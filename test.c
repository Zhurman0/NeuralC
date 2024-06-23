#include <stdio.h>
#include "NeuralC.h"
#include "mnist.h"


void printInfo(int i, Network *self, double *target, int len, int rightAns, int errAns, double error) {
    printf("Iteration: %d\n", i);
    printNetworkOutput(self);
    for (int i = 0; i < len; i++) {
        printf("%lf\t", target[i]);
    }
    printf("\n(Network: %d, Image: %d)\n", getIndexMaxOutput(self), train_label[i]);
    printf("Error: %lf\nRight answer: %d\tError answer: %d\n", error, rightAns, errAns);
    printf("Accuracy: %lf\n\n", (double)rightAns/(rightAns+errAns));
}

void train_network(Network *self, int iters) {
    double NetworkError = 0;
    unsigned int NetworkAnswer = 0;
    unsigned int rightAns = 0;
    unsigned int errAns = 0;

    for (int i = 0; i < iters; i++) {
        forwardPass(self, train_image[i]);
        NetworkAnswer = getIndexMaxOutput(self);
        
        double target[10] = {.01, .01, .01, .01, .01, .01, .01, .01, .01, .01};
        target[train_label[i]] = .9;

        NetworkError = calculateError(self, target);
        backPropagation(self, target, .001);

        if (train_label[i] == NetworkAnswer) rightAns++;
        else errAns++;

        if (!(i%500)) printInfo(i, self, target, 10, rightAns, errAns, NetworkError);
    }
}

int main(void) {
    Network *Nums = createNetwork(28*28, &layerLinear, &MSE); //CE (Cross Entropy) is also available, but it has errors in calculating the gradient
    addLayer(Nums, 16, &layerRelu); // avalive funcs for activation: Linear, Relu, LeakyRelu, Tanh, Sigmoid, Softmax
    addLayer(Nums, 16, &layerRelu);
    addLayer(Nums, 10, &layerSoftmax);

    load_mnist();

    train_network(Nums, 60000); // train network via mnist database
}