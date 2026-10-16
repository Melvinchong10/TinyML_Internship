import numpy as np

class FCLayerSim:
    def __init__(self, input_size=256, output_size=2, inner_lr=0.05, outer_lr = 0.005, supp_sz = 2, query_sz = 2, L2_lambda = 1e-4):
         self.W = np.random.randn(input_size, output_size) * 0.01
         self.b = np.zeros(output_size)
         self.input_size = input_size
         self.output_size = output_size
         self.inner_lr = inner_lr
         self.outer_lr = outer_lr
         self.supp_sz = supp_sz
         self.query_sz = query_sz
         self.L2_lambda = L2_lambda
         self.sim_index = 0

    def softmax(self, x):
         x = x - np.max(x)
         e = np.exp(x)
         return e / np.sum(e)

    def forward(self, x):
         self.x = x
         z = np.dot(x, self.W) + self.b
         self.out = self.softmax(z)
         return self.out

    def backward(self, y):
         error = self.out - y

         dW = np.outer(self.x, error)
         db = error

         self.W -= self.lr * dW
         self.b -= self.lr * db

    def FOMAML_forward(self, set, set_size):
        output = []
        for i in range(set_size):
            output.append(self.forward(set[i]))
        return np.array(output)

    def FOMAML_backward(self, output, set, ground_truth, loop_flag):
        weights_error = np.zeros((self.input_size, self.output_size))
        if(loop_flag == "inner"): 
            for i in range(self.supp_sz):
                supp_error = output[i] - ground_truth[i]
                weights_error += np.outer(set[i], supp_error) / self.supp_sz
                self.b -= self.inner_lr * supp_error/self.supp_sz

            self.W = self.W * (1 - ((self.inner_lr * self.L2_lambda)/self.supp_sz)) - self.inner_lr * weights_error
            
        else:
            for j in range(self.query_sz):
                query_error = output[j] - ground_truth[j]
                weights_error += np.outer(set[j], query_error) / self.query_sz
                self.initial_bias -= self.outer_lr * query_error/self.query_sz

            self.b = self.initial_bias
            self.W = self.initial_weights * (1 - ((self.outer_lr * self.L2_lambda)/self.query_sz)) - self.outer_lr * weights_error

    def L2(self, size):
         L2_sum = 0
         for i in range(self.W.shape[0]):
             for j in range(self.W.shape[1]):
                L2_sum += self.W[i][j]*self.W[i][j]
         return L2_sum / size

    def cross_entropy_loss(self, pred, real, size):
         result = 0.0
         epsilon = 1e-8
         for i in range(len(pred)):
            p = pred[i]
            r = real[i]
            for j in range(len(p)):
                if(r[j] == 1 and p[j] < epsilon):
                    result -= r[j]*np.log(epsilon)
                else:
                    result -= r[j]*np.log(p[j])

         return result / size

    def FOMAML_simulation_round(self, X_person_train_feat, X_nonperson_train_feat, y_person_train_oh, y_nonperson_train_oh, local_epoch, global_w = None):
        if(global_w is not None):
            self.W = global_w
            print("Updated FC Layer weights using the global aggregated weights")

        x_support = []
        y_support = []
        x_query = []
        y_query = []
        supp_loss = 0.0
        query_loss = 0.0
        avr_supp_loss = 0.0
        avr_supp_acc = 0.0

        #data collection
        x_person_support = X_person_train_feat[self.sim_index: (self.sim_index + self.supp_sz //2)]
        x_nonperson_support = X_nonperson_train_feat[self.sim_index: (self.sim_index + self.supp_sz //2)]
        y_person_support = y_person_train_oh[self.sim_index: (self.sim_index + self.supp_sz //2)]
        y_nonperson_support = y_nonperson_train_oh[self.sim_index: (self.sim_index + self.supp_sz //2)]

        for p_img, p_label, n_img, n_label in zip(
            x_person_support,
            y_person_support,
            x_nonperson_support,
            y_nonperson_support,
        ):
            x_support.append(p_img)
            y_support.append(p_label)

            x_support.append(n_img)
            y_support.append(n_label)

        x_support = np.array(x_support, dtype=np.float32)
        y_support = np.array(y_support, dtype=np.float32)

        self.sim_index += self.supp_sz//2

        x_person_query = X_person_train_feat[self.sim_index: (self.sim_index + self.query_sz //2)]
        x_nonperson_query = X_nonperson_train_feat[self.sim_index: (self.sim_index + self.query_sz //2)]
        y_person_query = y_person_train_oh[self.sim_index: (self.sim_index + self.query_sz //2)]
        y_nonperson_query = y_nonperson_train_oh[self.sim_index: (self.sim_index + self.query_sz //2)]
        
        for p_img, p_label, n_img, n_label in zip(
            x_person_query,
            y_person_query,
            x_nonperson_query,
            y_nonperson_query,
        ):
            x_query.append(p_img)
            y_query.append(p_label)

            x_query.append(n_img)
            y_query.append(n_label)

        x_query = np.array(x_query, dtype=np.float32)
        y_query = np.array(y_query, dtype=np.float32)

        self.sim_index += self.query_sz//2

        self.initial_weights = self.W.copy()
        self.initial_bias = self.b.copy()

        #before adaptation query acc + query loss
        correct = 0
        query_pred = self.FOMAML_forward(x_query, self.query_sz)
        l2_error = self.L2(self.query_sz) * self.L2_lambda
        error = self.cross_entropy_loss(query_pred, y_query, self.query_sz) + l2_error

        true = np.argmax(y_query, axis=1)
        pred = np.argmax(query_pred, axis=1)

        correct = np.sum(pred == true)
        query_before_acc = correct / len(true)
        query_before_loss = error

        for epoch in range(local_epoch):

            #inner
            correct = 0

            supp_pred = self.FOMAML_forward(x_support, self.supp_sz)
            l2_error = self.L2(self.supp_sz) * self.L2_lambda
            error = self.cross_entropy_loss(supp_pred, y_support, self.supp_sz) + l2_error
            self.FOMAML_backward(supp_pred, x_support, y_support, "inner")

            true = np.argmax(y_support, axis=1)
            pred = np.argmax(supp_pred, axis=1)

            correct = np.sum(pred == true)
            supp_acc = correct / len(true)
            supp_loss = error
            avr_supp_acc += supp_acc
            avr_supp_loss += supp_loss   
            print(f"Inner Epoch {epoch +1} | Supp Loss: {supp_loss:.4f} | Supp Acc: {supp_acc:.4f}")

        #outer
        correct = 0
        #after adaptation, query acc and query loss
        query_pred = self.FOMAML_forward(x_query, self.query_sz)
        l2_error = self.L2(self.query_sz) * self.L2_lambda
        error = self.cross_entropy_loss(query_pred, y_query, self.query_sz) + l2_error
        self.FOMAML_backward(query_pred, x_query, y_query, "outer")

        true = np.argmax(y_query, axis=1)
        pred = np.argmax(query_pred, axis=1)

        correct = np.sum(pred == true)
        query_after_acc = correct / len(true)
        query_after_loss = error

        avr_supp_acc = avr_supp_acc / local_epoch
        avr_supp_loss = avr_supp_loss /local_epoch

        print(f"Avg Support Loss: {avr_supp_loss:.4f} | Avg Support Acc: {avr_supp_acc:.4f}")
        print(f"Query Before Adapt Loss: {query_before_loss:.4f} | Query Before Adapt Acc: {query_before_acc:.4f}")
        print(f"Query After Adapt Loss: {query_after_loss:.4f} | Query After Adapt Acc: {query_after_acc:.4f}")

        print("Returning weights after FOMAML simulation round...")
        return self.W, avr_supp_acc, avr_supp_loss, query_before_acc, query_before_loss, query_after_acc, query_after_loss