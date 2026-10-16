// classes example
#include <iostream>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <math.h>		/* exp */
#include "FCLayer.h" 
// #include "FL_model_weights.h"
#include <cstring>
//#include "Particle.h"
#include <Arduino.h>
#include <cmath>
using namespace std;



FCLayer::FCLayer (){
	//bias = new double[1];
	//input_size = 0;
	//output_size = 0;
}

FCLayer::FCLayer (int input_sz, int output_sz, double scale, 
                    int zero_point, int batch, int support_sz, int query_sz, bool default_weight){
	input_size = input_sz;
	output_size = output_sz;
	support_size = support_sz;
	query_size = query_sz;
	batch_size = batch;
	quant_zero_point = zero_point;
	quant_scale = scale;

	bias = new double[output_size];
	for(int j = 0; j < output_size; j++) {
		if(default_weight){
			bias[j] = 0.0;
		}
		else{
			bias[j] = (double)rand()/(RAND_MAX);
		}
	}

	weights = new double*[input_size];
	for(int i = 0; i < input_size; i++) {
		weights[i] = new double[output_size];
		for(int j = 0; j < output_size; j++){
	    	if(default_weight){
	    		//initial_model_weights was 256x2 but now flattened
	    		//input_size = 256, output_size = 2
	    		weights[i][j] = 0.0;
	    	}else{
	    		if(rand()%2 == 0){
	    			weights[i][j] = -1*(double)rand()/(RAND_MAX);
	    		}else{
	    			weights[i][j] = (double)rand()/(RAND_MAX);
	    		}
	    	}
			
	    }
	}

	
	output_error_softmax = new double*[batch_size];
	for(int b = 0; b < batch_size; b++) {
	    output_error_softmax[b] = new double[output_size];
	}
	
	support_output_error_softmax = new double*[support_size];
	for(int b = 0; b < support_size; b++) {
	    support_output_error_softmax[b] = new double[output_size];
	}

	query_output_error_softmax = new double*[query_size];
	for(int b = 0; b < query_size; b++) {
	    query_output_error_softmax[b] = new double[output_size];
	}

}


void FCLayer::set_weights_bias(double *new_weights, double *new_bias){
	// Set weights
	for(int i = 0; i < input_size; ++i) {
		for (int j = 0; j < output_size; ++j){
			weights[i][j] = new_weights[i*output_size + j];
		}		
	}

	// Set bias
	for(int j = 0; j < output_size; j++) {
		bias[j] = new_bias[j];
	}
}

void softmax(double *input_output_data, int classes){
	double sum = 0;
	double max = -10000.0;
	for(int j = 0; j < classes; j++){
		if(input_output_data[j] > max){
			max = input_output_data[j];
		}
	}
	for(int j = 0; j < classes; j++){
		// minus a common max number to avoid infinity result when the input is too large
		input_output_data[j] = exp(input_output_data[j]-max);
		sum += input_output_data[j];
	}
	for(int j = 0; j < classes; j++){
		input_output_data[j] = input_output_data[j]/sum;
	}
}

// derivative of softmax
void softmax_prime(double *pred, double *result, int classes){
	for(int a = 0; a < classes; a++){
		result[a] = 0;
		for(int b = 0; b < classes; b++){
			if(a == b){
				result[a] += pred[a]*(1-pred[a]);
			}else{
				result[a] += -1*pred[a]*pred[b];
			}
		}
	}
}

void FCLayer::dequantize(int *input_data, double *input_float){
	//use double input [input_size]; when calling
	//input = (input - zero_point) * scale
	for(int i = 0; i < input_size; i++){
		input_float[i] = (input_data[i]-quant_zero_point)*quant_scale;
	}
}


void FCLayer::forward (double *input_float, double *output) {
	/*
	https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
	*/

	//self.output = np.dot(self.input, self.weights) + self.bias
	for(int b = 0; b < batch_size; b++){
		for(int j = 0; j < output_size; j++){
			output[b*output_size + j] = 0.0;
			for(int i = 0; i < input_size; i++){
				output[b*output_size + j] += input_float[b*input_size + i]*weights[i][j];
			}
			output[b*output_size + j] += bias[j];
		}
		//softmax to get probabilities
		//softmax(&output[b*input_size], output_size);
		softmax(&output[b*output_size], output_size);
	}
}

double L2(double **weight, int input_size, int output_size, int size){
	double sum = 0.0;
	for(int i = 0; i < input_size; i++){
		for(int j = 0; j < output_size; j++){
			sum += pow(weight[i][j],2);
		}
	}
	return sum / size;
}

void cross_entropy_prime(double *pred, int *real, double *result, int classes){
	for(int j = 0; j < classes; j++){
		result[j] = -1.0*real[j]/pred[j];
	}
}

double cross_entropy_loss(double *pred, int *real, int size, int classes){
	double result = 0.0;
	double epsilon = 0.0000001;

	for(int b = 0; b < size; b++){
		for(int j = 0; j < classes; j++){
			if(real[b*classes + j] == 1 and pred[b*classes + j] < epsilon){
				// use epsilon to prevent infinite result
				result -= real[b*classes + j]*log(epsilon);
			}else{
				result -= real[b*classes + j]*log(pred[b*classes + j]);
			}
		}
	}
	return result/(size);
}

void mse_prime(double *pred, int *real, double *result, int classes){
	for(int i = 0; i < classes; i++){
		result[i] += 2*(pred[i]-real[i]);
	}

}

double mse(double **pred, int **real, int size, int classes){
	double result = 0.0;

	for(int b = 0; b < size; b++){
		for(int j = 0; j < classes; j++){
			result += pow((pred[b][j]-real[b][j]), 2);
		}
	}
	return result/(classes*size);
}

void combined_ce_softmax_prime(double *pred, int *real, double *result, int classes){
	for(int i = 0; i < classes; i++){
		result[i] = pred[i] - real[i];
	}
}

double accuracy(double *pred, int *real, int size, int classes){
	double correct = 0.0;

	for(int b = 0; b < size; b++){
		for(int j = 0; j < classes; j++){
			if((pred[b*classes + j] >= 0.5 && real[b*classes + j] == 1)||(pred[b*classes + j] < 0.5 && real[b*classes + j] == 0)){
				correct += 1;
			}
		}
	}
	return correct/(classes*size);
}

void FCLayer::backward (double *output, int *ground_truth, 
						double *input_error, double *input_float,
						double learning_rate, double lambda) {

	//double **output_error = new double*[batch_size];

	for(int b = 0; b < batch_size; b++) {
	    //output_error[b] = new double[output_size];
	    //cross_entropy_prime(output[b], ground_truth[b], output_error[b], output_size);
	    //softmax_prime(output_error[b], output_error_softmax[b], output_size);
		
		// derivative of error (cross entropy loss) w.r.t outputs (after softmax) of last layer
	    combined_ce_softmax_prime(&output[b*output_size], &ground_truth[b*output_size], output_error_softmax[b], output_size);
	}

	for(int b = 0; b < batch_size; b++){
		//input_error = np.dot(output_error, self.weights.T)
		// batch_sizexinput_size = batch_sizexoutput_size dot output_sizexinput_size
	    for(int i = 0; i < input_size; i++) {
	    	input_error[b*input_size + i] = 0.0;
		    for(int j = 0; j < output_size; j++){
				// derivative of error w.r.t inputs
				input_error[b*input_size + i] += output_error_softmax[b][j]*weights[i][j]/batch_size;
		    }
		}
	}

	//weights_error = np.dot(self.input.T, output_error)
	double weights_error[input_size][output_size];
	
	for(int i = 0; i < input_size; i++) {
		for(int j = 0; j < output_size; j++){
			weights_error[i][j] = 0.0;
		}
	}

	for(int i = 0; i < input_size; i++) {
		for(int j = 0; j < output_size; j++){
			for(int b = 0; b < batch_size; b++){
				// derivative of error w.r.t weights
				weights_error[i][j] += input_float[b*input_size + i]*output_error_softmax[b][j]/batch_size;
		    }
		}
	}

	for(int i = 0; i < input_size; ++i) {
		for(int j = 0; j < output_size; j++){
			// cost function = loss function + L2 regularisation
			// step size = learning rate * derivative of cost function w.r.t weights
			// derivaitve of L2 regularisation w.r.t weights = (lambda/batch_size) * weights
			// step size = learning rate * derivative of L2 regularisation w.r.t weights + learning rate * derivative of error w.r.t weights
			this->weights[i][j] = this->weights[i][j]*(1-learning_rate*lambda/batch_size)-learning_rate*weights_error[i][j];
			// this->weights[i][j] = this->weights[i][j] - this->weights[i][j]*learning_rate*lambda/batch_size-learning_rate*weights_error[i][j];
		}
	}
	
	for(int j = 0; j < output_size; j++){
		for(int b = 0; b < batch_size; b++) {
			// derivative of error w.r.t bias = output_error_softmax
			// step size = learning rate * derivative of error w.r.t bias
			this->bias[j] -= learning_rate*output_error_softmax[b][j]/batch_size;
		}
	}
}

void FCLayer::cleanup(){
	for(int i = 0; i < input_size; i++) {
		delete [] weights[i];
	}
	for(int b = 0; b < batch_size; b++){
	    delete [] output_error_softmax[b];
	}
	delete [] output_error_softmax;
	delete[] weights;
	delete[] bias;
}

FCLayer::~FCLayer() { }

/*
void predict(double **input_float, FCLayer *model, double **output, int batch_size, int classes){
	model->forward(input_float, output);

	for(int b = 0; b < batch_size; b++){
		for(int j = 0; j < classes; j++){
			if(output[b][j] >= 0.5){
				output[b][j] = 1.0;
			}else{
				output[b][j] = 0.0;
			}
		}
	}
}*/

void FL_round_simulation(double *input_f, int *input_i, int *ground_truth, int local_episodes, 
						double learning_rate, FCLayer *model, double lambda, 
						bool verbose, bool local, bool unquantize){
	if(verbose == true){
		if(local == true){
			cout << "\tstarted sim\n";
		}else{
			Serial.print("\tstarted sim\n");
		}
	}
	// double **output = new double*[model->batch_size];
	// double **input_error = new double*[model->batch_size];
	// double **input_float = new double*[model->batch_size];
	
	double output[model->batch_size * model->output_size];
	double input_error[model->batch_size * model->input_size];
	double input_float[model->batch_size * model->input_size];


	if(verbose == true){
		if(local == true){
			cout << "\tallocated\n";
		}else{
			Serial.print("\tallocated\n");
		}
	}

	for(int b = 0; b < model->batch_size; b++) {
		for(int j = 0; j < model->output_size; j++){
			output[b*model->output_size + j] = 0.0;
		}

		for(int i = 0; i < model->input_size; i++){
			input_error[b*model->input_size + i] = 0.0;
		}
		
		if(unquantize == true){
			model->dequantize(&input_i[b*model->input_size], &input_float[b*model->input_size]);
		}else{
			for(int i = 0; i < model->input_size; i++){
				input_float[b*model->input_size + i] = input_f[b*model->input_size + i];
			}
		}
	}

	if(verbose == true){
		if(local == true){
			cout << "\tallocated part 2\n";
			cout << "\tinitial bias loaded " << model->bias[0] << " " << model->bias[1] << "\n";
			cout << "\tinitial output " << output[0*0] << " " << output[0*1] << "\n";
		}else{
			Serial.print("\tallocated\n");
		}
	}

	for(int epi = 0; epi < local_episodes; epi++){
		if(verbose == true){
			if(local == true){
				cout << "EPISODE " << epi << "\n";
				cout << "\tforward\n";
				cout << "\t\tbias loaded " << model->bias[0] << " " << model->bias[1] << "\n";
			}else{
				Serial.print("\tEPISODE");
        		Serial.println(epi);
//        		Serial.print("\tforward\n\t\t bias loaded");
//				Serial.print(model->bias[0], model->bias[1]);
			}
		}
		//forward
		model->forward(input_float, output);

		if(verbose == true){
			if(local == true){
				cout << "\tsoftmax " << output[0*0] << " " << output[0*1] << "\n";
			}else{
//				Serial.print("\tsoftmax");
//				Serial.println(output[0][0]);
//				Serial.println(output[0][1]);
			}
		}

		//calculate and print error
		double l2_error = lambda*L2(model->weights, model->input_size, model->output_size, model->batch_size)/2;
		double error = cross_entropy_loss(output, ground_truth, model->batch_size, model->output_size)
						+l2_error;
		double acc = accuracy(output, ground_truth, model->batch_size, model->output_size);
		Serial.print("accuracy: ");
		Serial.println(acc);

		//backward
		model->backward(output, ground_truth, input_error, input_float, learning_rate, lambda);

		if(verbose == true){
			if(local == true){
				cout << "\tbackward\n";
			}else{
				Serial.print("\tbackward\n");
			}
		}
		//reset input_error and output in forward and backward 
	}

	if(verbose == true){
		if(local == true){
			cout << "\tdone loop\n";
		}else{
			Serial.print("\tdone loop\n");
		}
	}
	
	/*
	for(int b = 0; b < model->batch_size; b++) {
		delete [] input_error[b];
		delete [] output[b];
		if(unquantize == true){
			delete [] input_float[b];
		}
	}
	delete [] input_error;
	delete [] output;
	delete [] input_float;*/

	if(verbose == true){
		if(local == true){
			cout << "\tdone de-allocation\n";
		}else{
			Serial.print("\tdone de-allocation\n");
		}
	}
}

void FCLayer::FOMAML_forward (double *input_float, double *output, int size) {
	/*
	https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
	*/

	//self.output = np.dot(self.input, self.weights) + self.bias
	for(int b = 0; b < size; b++){
		for(int j = 0; j < output_size; j++){
			output[b*output_size + j] = 0.0;
			for(int i = 0; i < input_size; i++){
				output[b*output_size + j] += input_float[b*input_size + i]*weights[i][j];
			}
			output[b*output_size + j] += bias[j];
		}
		
		Serial.print("Logits #");
		Serial.print(b);
		Serial.print(": [");
		for (int j = 0; j < output_size; j++) {
			Serial.print(output[b * output_size + j], 8);
			if (j < output_size - 1)
				Serial.print(", ");
		}
		Serial.println("]");
		
		//softmax to get probabilities
		//softmax(&output[b*input_size], output_size);
		softmax(&output[b*output_size], output_size);
	}
}

void FCLayer::FOMAML_backward (double *output, int *ground_truth, 
                            double *input_error, double *input_float,
                            double learning_rate, double lambda, double **initial_weights, double *initial_bias, 
							bool use_initial)
{
	//double **output_error = new double*[batch_size];
	//for outer loop
	if(use_initial == true)
	{
		for(int b = 0; b < this->query_size; b++) {
			//output_error[b] = new double[output_size];
			//cross_entropy_prime(output[b], ground_truth[b], output_error[b], output_size);
			//softmax_prime(output_error[b], output_error_softmax[b], output_size);
			
			// derivative of error (cross entropy loss) w.r.t outputs (after softmax) of last layer
			combined_ce_softmax_prime(&output[b*output_size], &ground_truth[b*output_size], query_output_error_softmax[b], output_size);
		}

		for(int b = 0; b < this->query_size; b++){
			//input_error = np.dot(output_error, self.weights.T)
			// batch_sizexinput_size = batch_sizexoutput_size dot output_sizexinput_size
			for(int i = 0; i < input_size; i++) {
				input_error[b*input_size + i] = 0.0;
				for(int j = 0; j < output_size; j++){
					// derivative of error w.r.t inputs
					input_error[b*input_size + i] += query_output_error_softmax[b][j]*weights[i][j]/this->query_size;
				}
			}
		}

		//weights_error = np.dot(self.input.T, output_error)
		double weights_error[input_size][output_size];

		for(int i = 0; i < input_size; i++) {
			for(int j = 0; j < output_size; j++){
				weights_error[i][j] = 0.0;
			}
		}

		for(int i = 0; i < input_size; i++) {
			for(int j = 0; j < output_size; j++){
				for(int b = 0; b < this->query_size; b++){
					// derivative of error w.r.t weights
					weights_error[i][j] += input_float[b*input_size + i]*query_output_error_softmax[b][j]/this->query_size;
				}
			}
		}
		
		/*
		double max_grad = 0.0;

		for (int i = 0; i < input_size; i++) {
			for (int j = 0; j < output_size; j++) {
				double g = fabs(weights_error[i][j]);
				if (g > max_grad)
					max_grad = g;
			}
		}
		
		Serial.print("Query Max gradient: ");
		Serial.println(max_grad, 10);
		*/

		for(int i = 0; i < input_size; ++i) {
			for(int j = 0; j < output_size; j++){
				// cost function = loss function + L2 regularisation
				// step size = learning rate * derivative of cost function w.r.t weights
				// derivaitve of L2 regularisation w.r.t weights = (lambda/batch_size) * weights
				// step size = learning rate * derivative of L2 regularisation w.r.t weights + learning rate * derivative of error w.r.t weights
				this->weights[i][j] = initial_weights[i][j]*(1-learning_rate*lambda/this->query_size)-learning_rate*weights_error[i][j];
				// this->weights[i][j] = this->weights[i][j] - this->weights[i][j]*learning_rate*lambda/batch_size-learning_rate*weights_error[i][j];
			}
		}

		for(int j = 0; j < output_size; j++){
			for(int b = 0; b < this->query_size; b++) {
				// derivative of error w.r.t bias = output_error_softmax
				// step size = learning rate * derivative of error w.r.t bias
				initial_bias[j] -= learning_rate*query_output_error_softmax[b][j]/this->query_size;
			}
			this->bias[j] = initial_bias[j];
		}
	}

	//for inner loop training
	else
	{

		for(int b = 0; b < this->support_size; b++) {
			//output_error[b] = new double[output_size];
			//cross_entropy_prime(output[b], ground_truth[b], output_error[b], output_size);
			//softmax_prime(output_error[b], output_error_softmax[b], output_size);
			
			// derivative of error (cross entropy loss) w.r.t outputs (after softmax) of last layer
			combined_ce_softmax_prime(&output[b*output_size], &ground_truth[b*output_size], support_output_error_softmax[b], output_size);
		}

		for(int b = 0; b < this->support_size; b++){
			//input_error = np.dot(output_error, self.weights.T)
			// batch_sizexinput_size = batch_sizexoutput_size dot output_sizexinput_size
			for(int i = 0; i < input_size; i++) {
				input_error[b*input_size + i] = 0.0;
				for(int j = 0; j < output_size; j++){
					// derivative of error w.r.t inputs
					input_error[b*input_size + i] += support_output_error_softmax[b][j]*weights[i][j]/this->support_size;
				}
			}
		}

		//weights_error = np.dot(self.input.T, output_error)
		double weights_error[input_size][output_size];

		for(int i = 0; i < input_size; i++) {
			for(int j = 0; j < output_size; j++){
				weights_error[i][j] = 0.0;
			}
		}

		for(int i = 0; i < input_size; i++) {
			for(int j = 0; j < output_size; j++){
				for(int b = 0; b < this->support_size; b++){
					// derivative of error w.r.t weights
					weights_error[i][j] += input_float[b*input_size + i]*support_output_error_softmax[b][j]/this->support_size;
				}
			}
		}

		/*
		double max_grad = 0.0;
		int max_i = 0;
		int max_j = 0;

		// Find the weight with the largest gradient
		for (int i = 0; i < input_size; i++) {
			for (int j = 0; j < output_size; j++) {
				double g = fabs(weights_error[i][j]);
				if (g > max_grad) {
					max_grad = g;
					max_i = i;
					max_j = j;
				}
			}
		}

		Serial.print("Support Max gradient: ");
		Serial.println(max_grad, 10);

		Serial.print("Max gradient location: [");
		Serial.print(max_i);
		Serial.print("][");
		Serial.print(max_j);
		Serial.println("]");

		Serial.print("Before W[");
		Serial.print(max_i);
		Serial.print("][");
		Serial.print(max_j);
		Serial.print("]: ");
		Serial.println(this->weights[max_i][max_j], 10);

		*/  
	
		for(int i = 0; i < input_size; ++i) {
			for(int j = 0; j < output_size; j++){
				// cost function = loss function + L2 regularisation
				// step size = learning rate * derivative of cost function w.r.t weights
				// derivaitve of L2 regularisation w.r.t weights = (lambda/batch_size) * weights
				// step size = learning rate * derivative of L2 regularisation w.r.t weights + learning rate * derivative of error w.r.t weights
				this->weights[i][j] = this->weights[i][j]*(1-learning_rate*lambda/this->support_size)-learning_rate*weights_error[i][j];
				// this->weights[i][j] = this->weights[i][j] - this->weights[i][j]*learning_rate*lambda/batch_size-learning_rate*weights_error[i][j];
			}
		}

		/*
		Serial.print("Gradient W[");
		Serial.print(max_i);
		Serial.print("][");
		Serial.print(max_j);
		Serial.print("]: ");
		Serial.println(weights_error[max_i][max_j], 10);

		Serial.print("After W[");
		Serial.print(max_i);
		Serial.print("][");
		Serial.print(max_j);
		Serial.print("]: ");
		Serial.println(this->weights[max_i][max_j], 10);
		*/

		for(int j = 0; j < output_size; j++){
			for(int b = 0; b < this->support_size; b++) {
				// derivative of error w.r.t bias = output_error_softmax
				// step size = learning rate * derivative of error w.r.t bias
				this->bias[j] -= learning_rate*support_output_error_softmax[b][j]/this->support_size;
			}
		}
	}
	
	return;
}


void FOMAML_round_simulation(double *input_f, int *support_set, int* query_set, int *support_ground_truth, int *query_ground_truth, int local_episodes, 
                        double innerloop_learning_rate, double outerloop_learning_rate, FCLayer *model, double lambda, 
                        bool verbose, bool local, bool unquantize)
{
	if(verbose == true)
	{
		if(local == true){
			cout << "\tstarted FOMAML sim\n";
		}else{
			Serial.print("\t[Nano] Started FOMAML sim\n");
		}
	}

	//double output[support_size * model->output_size];
	//double input_error[support_size * model->input_size];
	//double input_float[support_size * model->input_size];
	int max_size = 0;

	if(model->support_size >= model->query_size)
	{
		max_size = model->support_size;
	}

	else
	{
		max_size = model->query_size;
	}

	double* output = new double[max_size * model->output_size];
	double* input_error = new double[max_size * model->input_size];
	double* input_float = new double[max_size * model->input_size];
	double **initial_weights;
	double *initial_bias;

	double avr_supp_acc = 0.0;
	double avr_query_before_acc = 0.0;
	double avr_query_after_acc = 0.0;
	double avr_supp_loss = 0.0;
	double avr_query_before_loss = 0.0;
	double avr_query_after_loss = 0.0;
	//store theta values for outer loop update
	initial_bias = new double[model->output_size];
	for(int j = 0; j < model->output_size; j++) {
		initial_bias[j] = model->bias[j];
	}

	initial_weights = new double*[model->input_size];
	for(int i = 0; i < model->input_size; i++) {
		initial_weights[i] = new double[model->output_size];
		for(int j = 0; j < model->output_size; j++){
			initial_weights[i][j] = model->weights[i][j];
	    }
	}

	//query set before adaptation
	//initialize the arrays
	for(int b = 0; b < model->query_size; b++) {
		for(int j = 0; j < model->output_size; j++){
			output[b*model->output_size + j] = 0.0;
		}

		for(int i = 0; i < model->input_size; i++){
			input_error[b*model->input_size + i] = 0.0;
		}
		
		if(unquantize == true){
			model->dequantize(&query_set[b*model->input_size], &input_float[b*model->input_size]);
		}else{
			for(int i = 0; i < model->input_size; i++){
				input_float[b*model->input_size + i] = query_set[b*model->input_size + i];
			}
		}
	}

	if(verbose == true){
		if(local == true){
			cout << "\tallocated part 2\n";
			cout << "\tinitial bias loaded " << model->bias[0] << " " << model->bias[1] << "\n";
			cout << "\tinitial output " << output[0*0] << " " << output[0*1] << "\n";
		}else{
			Serial.print("\t[Nano] Allocated\n");
		}
	}

	//forward propagation for query set
	model->FOMAML_forward(input_float, output, model->query_size);

	if(verbose == true){
		if(local == true){
			cout << "\tsoftmax " << output[0*0] << " " << output[0*1] << "\n";
		}else{
			for(int k = 0; k < model->query_size; k++)
			{
				Serial.print("\t[Nano] Softmax query set #");
				Serial.print(k + 1);
				Serial.print(" : [");
				Serial.print(output[k*2]);
				Serial.print(",");
				Serial.print(output[(k*2) + 1]);
				Serial.println("]");
			}
		}
	}

	//calculate and print error
	double l2_error = lambda*L2(model->weights, model->input_size, model->output_size, model->query_size)/2;
	double error = cross_entropy_loss(output, query_ground_truth, model->query_size, model->output_size)
					+l2_error;
	double acc = accuracy(output, query_ground_truth, model->query_size, model->output_size);
	avr_query_before_loss = error;
	avr_query_before_acc = acc;
	Serial.print("[Nano] Query set before accuracy: ");
	Serial.println(acc);
	Serial.print("[Nano] Query set before loss: ");
	Serial.println(error, 8);

	for(int epi = 0; epi < local_episodes; epi++)
  	{
		
		if(verbose == true){
			if(local == true){
				cout << "EPISODE " << epi << "\n";
				cout << "\tforward\n";
				cout << "\t\tbias loaded " << model->bias[0] << " " << model->bias[1] << "\n";
			}else{
				Serial.print("\t[Nano] Inner Episode");
        		Serial.println(epi);
//        		Serial.print("\tforward\n\t\t bias loaded");
//				Serial.print(model->bias[0], model->bias[1]);
			}
		}

		//initialize the arrays
		for(int b = 0; b < model->support_size; b++) {
			for(int j = 0; j < model->output_size; j++){
				output[b*model->output_size + j] = 0.0;
			}

			for(int i = 0; i < model->input_size; i++){
				input_error[b*model->input_size + i] = 0.0;
			}
			
			if(unquantize == true){
				model->dequantize(&support_set[b*model->input_size], &input_float[b*model->input_size]);
			}else{
				for(int i = 0; i < model->input_size; i++){
					input_float[b*model->input_size + i] = support_set[b*model->input_size + i];
				}
			}
		}

		if(verbose == true){
			if(local == true){
				cout << "\tallocated part 2\n";
				cout << "\tinitial bias loaded " << model->bias[0] << " " << model->bias[1] << "\n";
				cout << "\tinitial output " << output[0*0] << " " << output[0*1] << "\n";
			}else{
				Serial.print("\t[Nano] Allocated\n");
			}
		}
			
		//forward propagation for support set
		model->FOMAML_forward(input_float, output, model->support_size);
		Serial.println("[Nano] Support set forwarded");

		if(verbose == true){
			if(local == true){
				cout << "\tsoftmax " << output[0*0] << " " << output[0*1] << "\n";
			}else{
				for(int k = 0; k < model->support_size; k++)
				{
					Serial.print("\t[Nano] Softmax supp set #");
					Serial.print(k + 1);
					Serial.print(" : [");
					Serial.print(output[k*2]);
					Serial.print(",");
					Serial.print(output[(k*2) + 1]);
					Serial.println("]");
				}
			}
		}

		//calculate and print error
		l2_error = lambda*L2(model->weights, model->input_size, model->output_size, model->support_size)/2;
		error = cross_entropy_loss(output, support_ground_truth, model->support_size, model->output_size)
						+l2_error;
		acc = accuracy(output, support_ground_truth, model->support_size, model->output_size);
		avr_supp_acc += acc;
		avr_supp_loss += error;
		Serial.print("[Nano] Support set accuracy: ");
		Serial.println(acc);
		Serial.print("[Nano] Support set loss: ");
		Serial.println(error, 8);

		//backward
		model->FOMAML_backward(output, support_ground_truth, input_error, input_float, innerloop_learning_rate, lambda, initial_weights, initial_bias, false);
		//Serial.println("Skipping support backward");
		if(verbose == true){
			if(local == true){
				cout << "\tbackward\n";
			}else{
				Serial.print("\t[Nano] Support backward\n");
			}
		}
		//reset input_error and output in forward and backward 
  	}

	//query set after adaptation
	//initialize the arrays
	for(int b = 0; b < model->query_size; b++) {
		for(int j = 0; j < model->output_size; j++){
			output[b*model->output_size + j] = 0.0;
		}

		for(int i = 0; i < model->input_size; i++){
			input_error[b*model->input_size + i] = 0.0;
		}
		
		if(unquantize == true){
			model->dequantize(&query_set[b*model->input_size], &input_float[b*model->input_size]);
		}else{
			for(int i = 0; i < model->input_size; i++){
				input_float[b*model->input_size + i] = query_set[b*model->input_size + i];
			}
		}
	}

	if(verbose == true){
		if(local == true){
			cout << "\tallocated part 2\n";
			cout << "\tinitial bias loaded " << model->bias[0] << " " << model->bias[1] << "\n";
			cout << "\tinitial output " << output[0*0] << " " << output[0*1] << "\n";
		}else{
			Serial.print("\t[Nano] Allocated\n");
		}
	}

	//forward propagation for query set
	model->FOMAML_forward(input_float, output, model->query_size);

	if(verbose == true){
		if(local == true){
			cout << "\tsoftmax " << output[0*0] << " " << output[0*1] << "\n";
		}else{
			for(int k = 0; k < model->query_size; k++)
			{
				Serial.print("\t[Nano] Softmax query set #");
				Serial.print(k + 1);
				Serial.print(" : [");
				Serial.print(output[k*2]);
				Serial.print(",");
				Serial.print(output[(k*2) + 1]);
				Serial.println("]");
			}
		}
	}

	//calculate and print error
	l2_error = lambda*L2(model->weights, model->input_size, model->output_size, model->query_size)/2;
	error = cross_entropy_loss(output, query_ground_truth, model->query_size, model->output_size)
					+l2_error;
    acc = accuracy(output, query_ground_truth, model->query_size, model->output_size);
	avr_query_after_loss = error;
	avr_query_after_acc = acc;
	Serial.print("[Nano] Query set after accuracy: ");
	Serial.println(acc);
	Serial.print("[Nano] Query set after loss: ");
	Serial.println(error, 8);
	//backward
	model->FOMAML_backward(output, query_ground_truth, input_error, input_float, outerloop_learning_rate, lambda, initial_weights, initial_bias, true);

	if(verbose == true){
		if(local == true){
			cout << "\tbackward\n";
		}else{
			Serial.print("\t[Nano] Query backward\n");
		}
	}
	//reset input_error and output in forward and backward 

	avr_supp_loss = avr_supp_loss / local_episodes; //average support loss
	avr_supp_acc = avr_supp_acc / local_episodes; //average support acc
	//avr_acc[1] = avr_acc[1]/  1; //average query acc

	Serial.print("[Nano] Average support loss: ");
	Serial.println(avr_supp_loss, 8);
	Serial.print("[Nano] Average query before loss: ");
	Serial.println(avr_query_before_loss, 8);
	Serial.print("[Nano] Average query after loss: ");
	Serial.println(avr_query_after_loss, 8);
	Serial.print("[Nano] Average support accuracy: ");
	Serial.println(avr_supp_acc);
	Serial.print("[Nano] Average query before accuracy: ");
	Serial.println(avr_query_before_acc);
	Serial.print("[Nano] Average query after accuracy: ");
	Serial.println(avr_query_after_acc);

	for (int i = 0; i < model->input_size; i++) {
		delete[] initial_weights[i];
	}
	delete[] initial_weights;
	delete[] initial_bias;
	delete[] output;
	delete[] input_error;
	delete[] input_float;
	
	if(verbose == true){
		if(local == true){
			cout << "\tdone de-allocation\n";
		}else{
			Serial.print("\t[Nano] Done de-allocation\n");
		}
	}

	return;
}

