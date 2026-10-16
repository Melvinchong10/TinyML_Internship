/*
 * Person Detection with On-Device Federated Learning
 * Platform: Coral Dev Board Micro (M7 core)
 *
 * Architecture:
 *   [Camera] → [TFLite Person Detect (frozen, feature extractor)]
 *            → [FCLayer (trainable head, 256→2)]
 *            → [FL local training per round]
 *
 * The TFLite model's penultimate layer (256-dim embedding, int8) is used
 * as the feature vector. The FCLayer trains on-device using backpropagation
 * with cross-entropy loss + L2 regularisation.
 *
 * FL round logic:
 *   1. Capture image → run TFLite inference → extract 256-dim embedding
 *   2. Accumulate a mini-batch of (embedding, label) pairs
 *   3. After FL_BATCH_SIZE samples: run FL_round_simulation() → local weight update
 *   4. Weights ready to be serialised / sent to aggregator via BLE or serial
 */

#include <Arduino.h>
#include <coral_micro.h>
#include <coralmicro_SD.h>
#include <coralmicro_camera.h>

#include "tensorflow/lite/micro/examples/person_detection/detection_responder.h"
#include "tensorflow/lite/micro/examples/person_detection/image_provider.h"
#include "tensorflow/lite/micro/examples/person_detection/main_functions.h"
#include "tensorflow/lite/micro/examples/person_detection/model_settings.h"
#include "libs/base/console_m7.h"
#include <stdlib.h>
#include "FCLayer.h"   // On-device FC layer with forward / backward / FL training

using namespace coralmicro::arduino;


// ─── TFLite setup ────────────────────────────────────────────────────────────
namespace {
  bool setup_success{false};

  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  const tflite::Model* model = nullptr;
  std::vector<uint8_t> model_data;
  constexpr char kModelPath[] = "/models/person_detection_model_mc.tflite";

  TfLiteTensor* input_tensor  = nullptr;
  TfLiteTensor* output_tensor = nullptr;   // <── penultimate embedding tensor
  std::unique_ptr<tflite::MicroInterpreter> interpreter = nullptr;
  tflite::MicroMutableOpResolver<5> micro_op_resolver;

  constexpr int kTensorArenaSize = 200 * 1024;
  STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);
}

// ─── FCLayer / FL hyper-parameters ───────────────────────────────────────────

/*
 * The person-detect model's embedding output is 256 int8 values.
 * After dequantisation they feed into a 256→2 FC layer (person / no-person).
 *
 * Adjust FL_BATCH_SIZE to trade RAM vs. update frequency.
 * Larger batches give stabler gradients but use more heap.
 */
constexpr int   FC_INPUT_SIZE   = 256;   // embedding dimension from TFLite
constexpr int   FC_OUTPUT_SIZE  = 2;     // classes: person=0, no-person=1
constexpr int   FL_BATCH_SIZE   = 1;     // samples per local FL round
//constexpr int   FL_LOCAL_EPOCHS = 5;     // SGD steps per FL round
//constexpr double FL_LR          = 0.01;  // learning rate
//constexpr double FL_LAMBDA      = 1e-4;  // L2 regularisation coefficient

constexpr int   SUPPORT_SET_SIZE   = 2;    // support set size
constexpr int   QUERY_SET_SIZE     = 2;    // query set size
constexpr double FOMAML_INNER_LR     = 0.01; // inner loop training rate
constexpr double FOMAML_OUTER_LR     = 0.001;  // outer loop training rate
constexpr double FOMAML_LAMBDA      = 1e-4;  // L2 regularisation coefficient
constexpr int   INNER_EPOCHS = 3;
/*
 * Quantisation params for the TFLite embedding tensor.
 * Read from interpreter->tensor(embedding_tensor_index)->params at runtime
 * and stored here after setup.
 */
double fc_quant_scale     = 1.0;
int    fc_quant_zero_point = 0;

FCLayer fc_model;   // The trainable head

int8_t local_tau = 1;

// ─── Batch accumulation buffers ───────────────────────────────────────────────
/*
 * Raw int8 embeddings collected before an FL round.
 * Stored as a flat array: [batch][embedding]
 */
//int    batch_input_i  [FL_BATCH_SIZE * FC_INPUT_SIZE];
//int    batch_labels   [FL_BATCH_SIZE * FC_OUTPUT_SIZE];  // one-hot
//int    batch_count    = 0;

//MAML parameters
int support_set_embeddings[SUPPORT_SET_SIZE * FC_INPUT_SIZE];
int query_set_embeddings[QUERY_SET_SIZE *FC_INPUT_SIZE];
int sample_embeddings[FC_INPUT_SIZE];
int support_set_labels   [SUPPORT_SET_SIZE * FC_OUTPUT_SIZE];
int query_set_labels   [QUERY_SET_SIZE * FC_OUTPUT_SIZE];   
double fc_input_f[FC_INPUT_SIZE];
double fc_output[FC_OUTPUT_SIZE];
int MAML_round = 0;

enum Label
{
    PERSON,
    NO_PERSON,
    INVALID
};

Label person_info = INVALID;
// ─── Label assignment ─────────────────────────────────────────────────────────
/*
 * For demonstration the ground-truth label is derived from the TFLite score.
 * Replace with a real labelling source (BLE command, button press, serial) for
 * a production FL deployment.
 *
 * person_score and no_person_score are uint8 from TFLite output tensor [0].
 * Label: person=[1,0], no-person=[0,1]
 */
/*void assign_label(uint8_t person_score, uint8_t no_person_score,
                  int* label_onehot) {
  if (person_score >= no_person_score) {
    label_onehot[0] = 1;  // person
    label_onehot[1] = 0;
  } else {
    label_onehot[0] = 0;  // no-person
    label_onehot[1] = 1;
  }
}
*/
Label serial_label(void)
{
    char buffer[16];

    while (Serial.available() == 0);

    size_t n = Serial.readBytesUntil('\n', buffer, sizeof(buffer) - 1);
    buffer[n] = '\0';

    if (n > 0 && buffer[n-1] == '\r')
        buffer[n-1] = '\0';

    if (strcmp(buffer, "y") == 0)
        return PERSON;

    if (strcmp(buffer, "n") == 0)
        return NO_PERSON;

    return INVALID;
}

void label_value(Label label)
{
  if(label == PERSON)
  {
    Serial.println("person");
  }

  else if(label == NO_PERSON)
  {
    Serial.println("no-person");
  }

  else
  {
    Serial.println("invalid");
  }
  return;
}

//using Serial Command for Supervised Learning
void assign_label(Label person_detail, int* label_onehot)
{
  if(person_detail == PERSON)
  {
    label_onehot[0] = 1;  // person
    label_onehot[1] = 0;
  }

  else 
  {
    label_onehot[0] = 0;  // no-person
    label_onehot[1] = 1;
  }
}

void RespondToDetectionModified(tflite::ErrorReporter* error_reporter,
                        Label person_info) {
  //TF_LITE_REPORT_ERROR(error_reporter, "person_info: %s",
                       //person_info.c_str());
  digitalWrite(PIN_LED_USER, person_info == PERSON ? HIGH : LOW);
}

void sendLocalWeights(void)
{
  Serial.println("BEGIN_WEIGHTS_TO_CLIENT");
  for(int i = 0; i < fc_model.input_size; i++)
  {
    for(int j = 0; j < fc_model.output_size; j++)
    {
      Serial.println(fc_model.weights[i][j], 10);
    }
  }
  Serial.println("END_WEIGHTS_TO_CLIENT");

  return;
}

void receiveGlobalWeights(void)
{
    bool receiving_w = false;
    int i = 0;
    int j = 0;
    //uint32_t last_recv_count = 0;
    char line[32];
    int pos = 0;
    //bool busy_blinked = false;
    //auto* console = coralmicro::ConsoleM7::GetSingleton();
    //console->debug_rx_count_ = 0;
    //console->debug_newline_count_ = 0;
    //console->debug_rx_available_ = 0;
    //console->debug_read_count_ = 0;
    //console->debug_read_bytes_ = 0;
    //console->ResetDebugRecvCounts();
    //console->ResetDebugEpComplete();
    Serial.println("[Coral] Starting..");
    Serial.println("REQ_WEIGHTS");
    Serial.println("Start Signal");

    while (true)
    { 
      /*
        uint32_t recv_count =
            console->GetDebugRecvResponseCount();

        if (recv_count != last_recv_count)
        {
            last_recv_count = recv_count;
            Serial.print("USB_DEBUG recv=");
            Serial.print(console->GetDebugRecvResponseCount());

            Serial.print(" handler=");
            Serial.print(console->GetDebugHandlerReturnCount());

            Serial.print(" rearm=");
            Serial.print(console->GetDebugRecvRearmCount());

            Serial.print(" status=");
            Serial.print(console->GetDebugRecvStatus());

            Serial.print(" epcomplete=0x");
            Serial.print(console->GetDebugEpComplete(), HEX);

            Serial.print(" available=");
            Serial.print(Serial.available());
            Serial.print(" rx_available=");
            Serial.print(console->debug_rx_available_);
            Serial.print(" read_count=");
            Serial.print(console->debug_read_count_);

            Serial.print(" read_bytes=");
            Serial.println(console->debug_read_bytes_);
            Serial.println(" DONE");

            digitalWrite(PIN_LED_USER, LOW); delay(300); digitalWrite(PIN_LED_USER, HIGH);
        }
        */
            while(Serial.available() > 0)
            {
            char c = Serial.read();
            if (c == '\n')
            {
                line[pos] = '\0';
                pos = 0;
                // Remove CR if CRLF
                int len = strlen(line);
                if (len > 0 && line[len - 1] == '\r')
                    line[len - 1] = '\0';

                if (strcmp(line, "BEGIN_WEIGHTS_TO_CORAL") == 0)
                {
                    receiving_w = true;
                    i = 0;
                    j = 0;

                    Serial.println("[Coral] Starting to receive weights from server..");
                }
                else if (strcmp(line, "END_WEIGHTS_TO_CORAL") == 0)
                {
                    Serial.println("[Coral] Weights successfully received");
                    return;
                }
                else if (receiving_w)
                {
                    double value = strtod(line, nullptr);

                    fc_model.weights[i][j] = value;
                    /*
                    Serial.print("[Coral] Weight[");
                    Serial.print(i);
                    Serial.print("][");
                    Serial.print(j);
                    Serial.print("]: ");
                    Serial.println(fc_model.weights[i][j], 10);
                    */
                    if (j == 1)
                    {
                        j = 0;
                        i++;
                    }
                    else
                    {
                        j++;
                    }

                    // One flash = one completed weight
                    //delay(500);
                    //digitalWrite(PIN_LED_USER, HIGH);
                }
            }
            else
            {
                if (pos < sizeof(line) - 1)
                {
                    line[pos++] = c;
                }
            }
        }
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}
// ─── Extract embedding from TFLite ───────────────────────────────────────────
/*
 * The person_detect model exposes TWO output tensors when not using TPU:
 *   [0] → final softmax scores (uint8, size 3 on MobileNet-based variants)
 *   [1] → penultimate embedding (int8, size 256)
 *
 * If your model only has one output you may need to modify it to expose the
 * embedding. Adjust kEmbeddingTensorIndex accordingly.
 *
 * For safety we always read from the tensor the interpreter exposes and
 * copy into the batch buffer with bounds checking.
 */
constexpr int kEmbeddingTensorIndex = 0;  // <── change if model differs

bool extract_embedding(int* dest) {
  TfLiteTensor* emb = interpreter->output_tensor(kEmbeddingTensorIndex);
  if (emb == nullptr) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Embedding tensor not found at index %d",
                         kEmbeddingTensorIndex);
    return false;
  }
  if (emb->dims->data[emb->dims->size - 1] != FC_INPUT_SIZE) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Embedding size mismatch: got %d, expected %d",
                         emb->dims->data[emb->dims->size - 1], FC_INPUT_SIZE);
    return false;
  }
  for (int i = 0; i < FC_INPUT_SIZE; i++) {
    dest[i] = static_cast<int>(emb->data.int8[i]);
  }
  return true;
}


// ─── FL round ─────────────────────────────────────────────────────────────────
//void run_fl_round() {
  //Serial.println("[FL] Starting local FL round...");
  //Serial.print("[FL] Batch size: ");
  //Serial.println(FL_BATCH_SIZE);

  /*
   * FL_round_simulation() signature:
   *   (input_f,   // float input — unused when unquantize=true
   *    input_i,   // int8 input batch, flat [batch * input_size]
   *    labels,    // one-hot int labels, flat [batch * output_size]
   *    episodes,  // local SGD steps
   *    lr, model, lambda, verbose, local=false, unquantize=true)
   *
   * We pass nullptr for input_f since we use the quantised path.
   */
  //FL_round_simulation(
    //nullptr,           // input_f  — not used (unquantize=true)
    //batch_input_i,     // int8 embeddings
    //batch_labels,      // one-hot ground truth
    //FL_LOCAL_EPOCHS,   // local training steps
    //FL_LR,
    //&fc_model,
    //FL_LAMBDA,
    //true,              // verbose
    //false,             // local=false → use Serial.print (on-device)
    //true               // unquantize=true → dequantize inside FL_round_simulation
  //);

  //Serial.println("[FL] Round complete. Weights updated.");

  // ── Optional: serialise weights to Serial for aggregator pickup ────────────
  // Uncomment the block below to print weights as CSV for a host aggregator.
  /*
  Serial.println("[FL] Weights:");
  for (int i = 0; i < FC_INPUT_SIZE; i++) {
    for (int j = 0; j < FC_OUTPUT_SIZE; j++) {
      Serial.print(fc_model.weights[i][j], 6);
      if (j < FC_OUTPUT_SIZE - 1) Serial.print(",");
    }
    Serial.println();
  }
  Serial.println("[FL] Bias:");
  for (int j = 0; j < FC_OUTPUT_SIZE; j++) {
    Serial.print(fc_model.bias[j], 6);
    if (j < FC_OUTPUT_SIZE - 1) Serial.print(",");
  }
  Serial.println();
  */
//}

void run_FOMAML_round()
{
  Serial.println("[Coral] Starting local FOMAML round...");
  Serial.print("[Coral] Support set size: ");
  Serial.println(SUPPORT_SET_SIZE);
  Serial.print("[Coral] Query set size: ");
  Serial.println(QUERY_SET_SIZE);

  FOMAML_round_simulation(
    nullptr,           // input_f  — not used (unquantize=true)
    support_set_embeddings, // int8 support embeddings
    query_set_embeddings, // int8 query embeddings
    support_set_labels,  // one-hot ground truth
    query_set_labels, // one-hot ground truth
    INNER_EPOCHS,// local training steps
    FOMAML_INNER_LR, //inner training learning rate
    FOMAML_OUTER_LR, //outer training learning rate
    &fc_model,
    FOMAML_LAMBDA,
    true,              // verbose
    false,             // local=false → use Serial.print (on-device)
    true               // unquantize=true → dequantize inside FL_round_simulation
  );

  /*
for (int i = 0; i < QUERY_SET_SIZE; i++) {
    fc_model.dequantize(&query_set_embeddings[i * FC_INPUT_SIZE], fc_input_f);
    fc_model.FOMAML_forward(fc_input_f, fc_output, 1);

    int pred = (fc_output[0] > fc_output[1]) ? 0 : 1;
    int gt = (query_set_labels[i * FC_OUTPUT_SIZE] == 1) ? 0 : 1;

    Serial.print("Query ");
    Serial.print(i);
    Serial.print(" GT=");
    Serial.print(gt);
    Serial.print(" Pred=");
    Serial.print(pred);
    Serial.print(" Prob=[");
    Serial.print(fc_output[0], 6);
    Serial.print(", ");
    Serial.print(fc_output[1], 6);
    Serial.println("]");
}
    */
    Serial.println("[Coral] Local round complete. Weights updated.");

    return;
}

// ─── setup() ──────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  pinMode(PIN_LED_STATUS, OUTPUT);
  digitalWrite(PIN_LED_STATUS, HIGH);
  //delay for Serial Setup
  delay(15000);
  Serial.println("Person Detection + On-Device FOMAML Supervised Training");

  pinMode(PIN_LED_USER, OUTPUT);

  // ── Load TFLite model from SD ──────────────────────────────────────────────
  SD.begin();
  TF_LITE_REPORT_ERROR(error_reporter, "Loading TFLite model...");
  if (!SD.exists(kModelPath)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Model file not found on SD card");
    return;
  }

  SDFile model_file = SD.open(kModelPath);
  uint32_t model_size = model_file.size();
  model_data.resize(model_size);
  if (model_file.read(model_data.data(), model_size) != model_size) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to read model");
    return;
  }

  model = tflite::GetModel(model_data.data());
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model schema version %d != supported %d",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // ── Build interpreter ─────────────────────────────────────────────────────
  //micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  //micro_op_resolver.AddReshape();
  //micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddConcatenation();
  //micro_op_resolver.AddLogistic();
  //micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddPad();
  micro_op_resolver.AddMean();
  interpreter = std::make_unique<tflite::MicroInterpreter>(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  input_tensor  = interpreter->input_tensor(0);
  //output_tensor = interpreter->output_tensor(1);  // scores tensor

  // ── Read quantisation params from embedding tensor ─────────────────────────
   TfLiteTensor* emb_tensor = interpreter->output_tensor(kEmbeddingTensorIndex);
   if (emb_tensor != nullptr) {
     fc_quant_scale      = static_cast<double>(emb_tensor->params.scale);
     fc_quant_zero_point = emb_tensor->params.zero_point;
     Serial.print("[FC] Embedding quant scale=");
     Serial.print(fc_quant_scale, 6);
     Serial.print("  zero_point=");
     Serial.println(fc_quant_zero_point);
   } else {
     Serial.println("[FC] Warning: could not read embedding tensor params, using defaults.");
   }

  // ── Initialise FCLayer ────────────────────────────────────────────────────
  /*
   * FCLayer(input_sz, output_sz, quant_scale, quant_zero_point,
   *         batch_size, default_weight)
   *
   * default_weight=false → random weight init (Xavier-like via rand())
   * Set to true if loading pre-trained weights via set_weights_bias().
   */
  fc_model = FCLayer(FC_INPUT_SIZE, FC_OUTPUT_SIZE,
                      fc_quant_scale, fc_quant_zero_point,
                     FL_BATCH_SIZE, SUPPORT_SET_SIZE, QUERY_SET_SIZE, false);

  Serial.println("[FC] FCLayer initialised.");
  Serial.print("[FC] Parameters: input=");
  Serial.print(FC_INPUT_SIZE);
  Serial.print("  output=");
  Serial.print(FC_OUTPUT_SIZE);
  Serial.print("  batch=");
  Serial.println(FL_BATCH_SIZE);

  // ── Start camera ──────────────────────────────────────────────────────────
  if (Camera.begin(kNumRows, kNumCols,
                   coralmicro::CameraFormat::kY8,
                   coralmicro::CameraFilterMethod::kBilinear,
                   coralmicro::CameraRotation::k270,
                   true) != CameraStatus::SUCCESS) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to start camera.");
    return;
  }

  setup_success = true;

  Serial.println("[Setup] Ready. Collecting samples for FOMAML training...");

}

// ─── loop() ───────────────────────────────────────────────────────────────────
void loop() {
  if (!setup_success) {
    Serial.println("Setup failed — cannot run.");
    delay(1000);
    return;
  }

  if (MAML_round == 0)
  {
      Serial.println("[Coral] Flushing first three frames...");

      for (int i = 0; i < 3; i++)
      {
          if (GetImage(error_reporter,
                      kNumRows,
                      kNumCols,
                      kNumChannels,
                      input_tensor->data.int8) != kTfLiteOk)
          {
              TF_LITE_REPORT_ERROR(error_reporter,
                                  "Image capture for flushing failed.");
              return;
          }
      }

      Serial.println("[Coral] Flush complete.");
      digitalWrite(PIN_LED_USER, HIGH);
      //type anything to start
    do
      {
        Serial.print("[Coral] Test: "); 
        person_info = serial_label();
        label_value(person_info);
    } while (person_info == INVALID);
      //Send ready signal to connect to server
    Serial.println("CONNECT");
  }

  Serial.print("[Coral] FOMAML Simulation round ");
  Serial.println(MAML_round);
  if(MAML_round % local_tau == 0)
  {
    //retrieve global weights from server
    Serial.println("[Coral] Receiving initial global weights from server");
    receiveGlobalWeights();
  }

  //Data collection for MAML Support Set
  for(int i = 0; i < SUPPORT_SET_SIZE; i++)
  { 
    Serial.print("[Coral] Support set batch ");
    Serial.println((i+1)); 

    //get label for supervised training
    do
    {
      Serial.print("[Coral] Support set label: "); 
      person_info = serial_label();
      label_value(person_info);
    } while (person_info == INVALID);
    

    // ── 1. Capture image into TFLite input tensor ──────────────────────────────
    if (GetImage(error_reporter, kNumRows, kNumCols, kNumChannels,
                input_tensor->data.int8) != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "[Coral] Image capture for support set failed.");
      return;
    }
    Serial.println("[Coral] Image captured! Hold camera still for 1 second.");
    delay(1000);
    Serial.println("[Coral] Done!");
    // ── 2. Run TFLite inference (frozen backbone) ──────────────────────────────
    if (interpreter->Invoke() != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "[Coral] Invoke failed for support set.");
      return;
    }

    RespondToDetectionModified(error_reporter, person_info);
    
    // ── 4. Extract 256-dim int8 embedding from penultimate layer ───────────────
    int* emb_dest = &support_set_embeddings[i * FC_INPUT_SIZE];
    if (!extract_embedding(emb_dest)) {
      Serial.println("[Coral] Skipping sample — embedding extraction for support set failed.");
      return;
    }
    Serial.print("[Coral] Embeddings for support set batch ");
    Serial.print((i+1));
    Serial.println(" collected");
    int* supp_label_dest = &support_set_labels[i * FC_OUTPUT_SIZE];
    assign_label(person_info, supp_label_dest);
  }

  //Data collection for MAML Query Set
  for(int j = 0; j < QUERY_SET_SIZE; j++)
  {
    Serial.print("[Coral] Query set batch ");
    Serial.println((j+1));

    //get label for supervised training
    do
    {
      Serial.print("[Coral] Query set label: ");
      person_info = serial_label();
      label_value(person_info);
    } while (person_info == INVALID);
    
    // ── 1. Capture image into TFLite input tensor ──────────────────────────────
    if (GetImage(error_reporter, kNumRows, kNumCols, kNumChannels,
                input_tensor->data.int8) != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "[Coral] Image capture for query set failed.");
      return;
    }
    Serial.println("[Coral] Image captured! Hold camera still for 1 second.");
    delay(1000);
    Serial.println("[Coral] Done!");
    // ── 2. Run TFLite inference (frozen backbone) ──────────────────────────────
    if (interpreter->Invoke() != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "[Coral] Invoke failed for query set.");
      return;
    }

    RespondToDetectionModified(error_reporter, person_info);
    
    // ── 4. Extract 256-dim int8 embedding from penultimate layer ───────────────
    int* emb_dest = &query_set_embeddings[j * FC_INPUT_SIZE];
    if (!extract_embedding(emb_dest)) {
      Serial.println("[Coral] Skipping sample — embedding extraction for query set failed.");
      return;
    }

    Serial.print("[Coral] Embeddings for query set batch ");
    Serial.print((j+1));
    Serial.println(" collected");
    int* query_label_dest = &query_set_labels[j * FC_OUTPUT_SIZE];
    assign_label(person_info, query_label_dest);
  }

  run_FOMAML_round();
  MAML_round++;


  if(MAML_round % local_tau == 0)
  {
    sendLocalWeights();
  }
  /*
  Serial.println("\n===== SUPPORT vs QUERY DISTANCES =====");
  for (int s = 0; s < SUPPORT_SET_SIZE; s++)
  {
      bool supportIsPerson =
          (support_set_labels[s * FC_OUTPUT_SIZE] == 1);

      for (int q = 0; q < QUERY_SET_SIZE; q++)
      {
          bool queryIsPerson =
              (query_set_labels[q * FC_OUTPUT_SIZE] == 1);

          double dist = 0.0;

          for (int i = 0; i < FC_INPUT_SIZE; i++)
          {
              double d =
                  (double)support_set_embeddings[s * FC_INPUT_SIZE + i] -
                  (double)query_set_embeddings[q * FC_INPUT_SIZE + i];

              dist += d * d;
          }

          dist = sqrt(dist);

          Serial.print("S");
          Serial.print(s);
          Serial.print(" (");
          Serial.print(supportIsPerson ? "Person" : "Wall");
          Serial.print(") <-> Q");
          Serial.print(q);
          Serial.print(" (");
          Serial.print(queryIsPerson ? "Person" : "Wall");
          Serial.print(") = ");
          Serial.println(dist);
      }
  }
  */
  //run tests for accuracy without training every n * 50 rounds
  /*
  if((MAML_round % 50) == 0)
  {
    int correct = 0;
    Serial.print("FOMAML round #");
    Serial.print(MAML_round);
    Serial.println(" reached, starting to take samples for accuracy test.");
    for(int s = 0; s < 10; s++)
    {
      //get label for accuracy check supervised training
      do
      {
        Serial.print("Sample set label: "); 
        person_info = serial_label();
        label_value(person_info);
      } while (person_info == INVALID);

      // ── 1. Capture image into TFLite input tensor ──────────────────────────────
      if (GetImage(error_reporter, kNumRows, kNumCols, kNumChannels,
                  input_tensor->data.int8) != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "Image capture for sample failed.");
        return;
      }

      // ── 2. Run TFLite inference (frozen backbone) ──────────────────────────────
      if (interpreter->Invoke() != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed for sample set.");
        return;
      }

      RespondToDetectionModified(error_reporter, person_info);
      
      // ── 4. Extract 256-dim int8 embedding from penultimate layer ───────────────
      int* emb_dest = sample_embeddings;
      if (!extract_embedding(emb_dest)) {
        Serial.println("[FC] Skipping sample — embedding extraction for query set failed.");
        return;
      }

      Serial.print("Embeddings for sample ");
      Serial.print((s+1));
      Serial.println(" collected");
      fc_model.dequantize(emb_dest, fc_input_f);
      fc_model.FOMAML_forward(fc_input_f, fc_output, 1); //only one sample per iteration

      Serial.print("[FC] person_prob=");
      Serial.print(fc_output[0], 4);
      Serial.print("  no_person_prob=");
      Serial.println(fc_output[1], 4);

      if((person_info == PERSON && fc_output[0] > fc_output[1]) || (person_info == NO_PERSON && fc_output[1] > fc_output[0]))
      {
        correct++;
      }

    }

    Serial.print("Accuracy: ");
    Serial.print(correct);
    Serial.println("/10");

  }
  */
  /*
  if(((MAML_round % 50) == 0 && MAML_round <= 250) || ((MAML_round % 10) == 0 && MAML_round > 250))
  {
    Serial.print("FOMAML round #");
    Serial.print(MAML_round);
    Serial.println(" reached, testing model for 20 seconds.");
    Serial.println("Simulations begins in: 5");
    delay(1000);
    Serial.println("4");
    delay(1000);
    Serial.println("3");
    delay(1000);
    Serial.println("2");
    delay(1000);
    Serial.println("1");
    delay(1000);
    Serial.println("Testing Started.");
    delay(1000);

    for(int s = 0; s < 200; s++)
    {

      // ── 1. Capture image into TFLite input tensor ──────────────────────────────
      if (GetImage(error_reporter, kNumRows, kNumCols, kNumChannels,
                  input_tensor->data.int8) != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "Image capture for sample failed.");
        return;
      }

      // ── 2. Run TFLite inference (frozen backbone) ──────────────────────────────
      if (interpreter->Invoke() != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed for sample set.");
        return;
      }
     
      // ── 4. Extract 256-dim int8 embedding from penultimate layer ───────────────
      int* emb_dest = sample_embeddings;
      if (!extract_embedding(emb_dest)) {
        Serial.println("[FC] Skipping sample — embedding extraction for query set failed.");
        return;
      }

      Serial.print("Embeddings for sample ");
      Serial.print((s+1));
      Serial.println(" collected");
      fc_model.dequantize(emb_dest, fc_input_f);
      fc_model.FOMAML_forward(fc_input_f, fc_output, 1); //only one sample per iteration

      Serial.print("[FC] person_prob=");
      Serial.print(fc_output[0], 4);
      Serial.print("  no_person_prob=");
      Serial.println(fc_output[1], 4);

      if(fc_output[0] > fc_output[1])
      {
        digitalWrite(PIN_LED_USER, HIGH);
        Serial.print("Person Status[FOMAML]: ");
        Serial.println("Person");
      }

      else
      {
        digitalWrite(PIN_LED_USER, LOW);
        Serial.print("Person Status[FOMAML]: ");
        Serial.println("Non-Person");
      }
      delay(100);
    }
    Serial.println("Testing period finished. Resuming training.");

  }
  */

  delay(100);  // ~10 fps capture rate; reduce for faster sampling
}
