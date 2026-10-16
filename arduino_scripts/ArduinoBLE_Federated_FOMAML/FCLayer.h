#ifndef NN_FC_H
#define NN_FC_H


class FCLayer {
    public: 
        int input_size, output_size, batch_size, support_size, query_size, quant_zero_point;
        double quant_scale;
        double *bias;
        double **weights;
        double **output_error_softmax;
        double **support_output_error_softmax;
        double **query_output_error_softmax;

        FCLayer();
        FCLayer (int input_sz, int output_sz, double scale, 
                    int zero_point, int batch, int support_sz, int query_sz, bool default_weight);
        void forward (double *input_float, double *output);
        void backward (double *output, int *ground_truth, 
                            double *input_error, double *input_float,
                            double learning_rate, double lambda);
        void FOMAML_forward (double *input_float, double *output, int size);
        void FOMAML_backward (double *output, int *ground_truth, 
                            double *input_error, double *input_float,
                            double learning_rate, double lambda, double **initial_weights, double *inital_bias
                        , bool use_initial);
        void dequantize(int *input_data, double *input_float);
        void cleanup();
        ~FCLayer();
        void set_weights_bias(double *new_weights, double *new_bias);
};

// void FL_round_simulation(double **input_float, int **ground_truth, int local_episodes, 
//                         double learning_rate, FCLayer *model, bool verbose, bool local);
void FL_round_simulation(double *input_f, int *input_i, int *ground_truth, int local_episodes, 
                        double learning_rate, FCLayer *model, double lambda, 
                        bool verbose, bool local, bool unquantize);
						
                        
void FOMAML_round_simulation(double *input_f, int *support_set, int* query_set, int *support_ground_truth, int *query_ground_truth, int local_episodes, 
                        double innerloop_learning_rate, double outerloop_learning_rate, FCLayer *model, double lambda, 
                        bool verbose, bool local, bool unquantize);
#endif
