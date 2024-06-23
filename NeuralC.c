#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>


//structures
struct Neuron {
    double output, deriv, bias;
    double *weights, *inputs, *gradients;
    unsigned int numCons;
};

struct Layer {
    struct Neuron *neurons;
    unsigned int numNeurons;
    void (*actFunc)(struct Layer *);
    void (*derivActFunc)(struct Layer *);
};

typedef struct Network {
    struct Layer *layers;
    unsigned int numLayers;
    double (*errFunc)(struct Network *, double *);
    void (*derivErrorFunc)(struct Network *, double *);
} Network;


// instruments
double __randDouble(double min, double max) {
    double range = max - min;
    double div = (0x7fff / range);
    return min + (rand() / div);
}

double __neuronLinear(struct Neuron *neuron) {
    double sum = neuron->bias;
    for (int i = 0; i < neuron->numCons; i++) {
        sum += neuron->weights[i] * neuron->inputs[i];
    }
    return sum;
}




//activation functions
void layerLinear(struct Layer *layer) {
    for (int i = 0; i < layer->numNeurons; i++) {
        struct Neuron *neuron = &layer->neurons[i];

        neuron->output = neuron->bias;
        for (int i = 0; i < neuron->numCons; i++) {
            neuron->output += neuron->weights[i] * neuron->inputs[i];
        }
    }
}

void layerRelu(struct Layer *layer) {
    layerLinear(layer);
    
    for (int i = 0; i < layer->numNeurons; i++) {
        double output = layer->neurons[i].output;
        layer->neurons[i].output = output > 0 ? output : 0;
    }
}

void layerLeakyRelu(struct Layer *layer) {
    layerLinear(layer);
    
    for (int i = 0; i < layer->numNeurons; i++) {
        double output = layer->neurons[i].output;
        layer->neurons[i].output = output > 0 ? output : output * .01;
    }
}

void layerSigmoid(struct Layer *layer) {
    layerLinear(layer);

    for (int i = 0; i < layer->numNeurons; i++) {
        double output = layer->neurons[i].output;
        layer->neurons[i].output = 1.0 / (1.0 + exp(-output));
    }    
}

void layerTanh(struct Layer *layer) {
    layerLinear(layer);

    for (int i = 0; i < layer->numNeurons; i++) {
        double output = layer->neurons[i].output;
        layer->neurons[i].output = 2.0 / (1.0 + exp(-2.0 * output)) - 1.0;
    }   
}

void layerSoftmax(struct Layer *layer) {
    layerLinear(layer);

    double exp_sum = 0;
    for (int i = 0; i < layer->numNeurons; i++) {
        exp_sum += exp(layer->neurons[i].output);
    }

    for (int i = 0; i < layer->numNeurons; i++) {
        double output = layer->neurons[i].output;
        layer->neurons[i].output = exp(output) / exp_sum;
    }
}


//derivative functions
void __layerLinearDeriv(struct Layer *layer) {
    for (int i = 0; i < layer->numNeurons; i++) {
        layer->neurons[i].deriv = 1.0;
    }
}

void __layerReluDeriv(struct Layer *layer) {
    for (int i = 0; i < layer->numNeurons; i++) {
        double output = layer->neurons[i].output;
        layer->neurons[i].deriv = output > 0 ? 1.0 : 0.0;
    }
}

void __layerLeakyReluDeriv(struct Layer *layer) {
    for (int i = 0; i < layer->numNeurons; i++) {
        double output = layer->neurons[i].output;
        layer->neurons[i].deriv = output > 0 ? 1.0 : 0.01;
    }
}

void __layerSigmoidDeriv(struct Layer *layer) {
    for (int i = 0; i < layer->numNeurons; i++) {
        double output = layer->neurons[i].output;
        layer->neurons[i].deriv = output * (1.0 - output);
    }
}

void __layerTanhDeriv(struct Layer *layer) {
    for (int i = 0; i < layer->numNeurons; i++) {
        double output = layer->neurons[i].output;
        layer->neurons[i].deriv = 1.0 - output * output;
    }
}

void __layerSoftmaxDeriv(struct Layer *layer) {
    for (int i = 0; i < layer->numNeurons; i++) {
        double output = layer->neurons[i].output;
        layer->neurons[i].deriv = output * (!i ? 1 - output : -output);
    }
}





//errors
double MSE(Network *self, double *targets) {
    struct Layer *lastLayer = &self->layers[self->numLayers - 1];

    double error = 0;
    for (int j = 0; j < lastLayer->numNeurons; j++) {
        double div = targets[j] - lastLayer->neurons[j].output;
        error += div*div;
    }

    return error / lastLayer->numNeurons;
}

void __MSEGradient(Network *self, double *targets) {
    struct Layer *lastLayer = &self->layers[self->numLayers - 1];
    lastLayer->derivActFunc(lastLayer);

    for (int i = 0; i < lastLayer->numNeurons; i++) {
        struct Neuron *neuron = &lastLayer->neurons[i];
        double output = neuron->output;
        double delta = -targets[i] + output;

        for (int j = 0; j < neuron->numCons; j++) {
            neuron->gradients[j] = delta * neuron->inputs[j];
        }
        neuron->gradients[neuron->numCons] = delta * neuron->deriv;
    }
}


double CE(Network *self, double *targets) { // Cross-Entropy
    struct Layer *lastLayer = &self->layers[self->numLayers - 1];

    double entropy = 0;
    for (int i = 0; i < lastLayer->numNeurons; i++) {
        entropy += targets[i] * log(lastLayer->neurons[i].output);
    }

    return -entropy;
}

void __CEGradient(Network *self, double *targets) {
    struct Layer *lastLayer = &self->layers[self->numLayers - 1];
    lastLayer->derivActFunc(lastLayer);

    for (int i = 0; i < lastLayer->numNeurons; i++) {
        struct Neuron *neuron = &lastLayer->neurons[i];
        double linear = __neuronLinear(neuron);
        double mainGradient = 0;

        for (int j = 0; j < lastLayer->numNeurons; j++) {
            struct Neuron *otherNeuron = &lastLayer->neurons[j];
            mainGradient += (targets[j] / otherNeuron->output); //* (otherNeuron->deriv / linear);
        }

        for (int j = 0; j < neuron->numCons; j++) {
            neuron->gradients[j] = mainGradient * neuron->inputs[j];
        }
        neuron->gradients[neuron->numCons] = mainGradient * neuron->deriv;
    }
}


// calculate gradients for hidden layers
void __hiddenLayersGradient(Network *self) {
    for (int lyr_i = self->numLayers - 2; lyr_i > 1; lyr_i--) {
        struct Layer *layer = &self->layers[lyr_i];
        struct Layer *nextLayer = &self->layers[lyr_i + 1];

        layer->derivActFunc(layer);
        nextLayer->derivActFunc(nextLayer);

        for (int i = 0; i < layer->numNeurons; i++) {
            struct Neuron *neuron = &layer->neurons[i]; 
            double gradientSum = 0;

            for (int j = 0; j < nextLayer->numNeurons; j++) {
                double *nextGradients = nextLayer->neurons[j].gradients;
                gradientSum += nextGradients[i] * nextLayer->neurons[j].weights[i];
            }

            
            double delta = gradientSum * nextLayer->neurons[i].deriv;
            for (int k = 0; k < neuron->numCons; k++) {
                neuron->gradients[k] = delta * neuron->inputs[k];
            }
            neuron->gradients[neuron->numCons] = delta * neuron->deriv;
        }
    }
}

void __applyGradients(Network *self, double learningRate) {
    for (int i = 1; i < self->numLayers; i++) {
        struct Layer *layer = &self->layers[i];

        for (int j = 0; j < layer->numNeurons; j++) {
            struct Neuron *neuron = &layer->neurons[j];

            neuron->bias -= learningRate * neuron->gradients[neuron->numCons];
            for (int k = 0; k < neuron->numCons; k++) {
                neuron->weights[k] -= learningRate * neuron->gradients[k];
            }
        }
    }
}



void __upgradeInputs(Network *self) {
    for (int i = 1; i < self->numLayers; i++) {
        struct Layer *layer = &self->layers[i];

        for (int j = 0; j < layer->numNeurons; j++) {
            struct Neuron *neuron = &layer->neurons[j];

            for (int k = 0; k < neuron->numCons; k++) {
                neuron->inputs[k] = self->layers[i - 1].neurons[k].output;
            }
        }
        layer->actFunc(layer);
    }
}

void forwardPass(Network *self, double *data) {
    struct Layer *startLayer = &self->layers[0];
    for (int i = 0; i < startLayer->numNeurons; i++) {
        startLayer->neurons[i].output = data[i];
    }
    __upgradeInputs(self);
}

void backPropagation(Network *self, double *target, double learningRate) {
    self->derivErrorFunc(self, target);
    __hiddenLayersGradient(self);
    __applyGradients(self, learningRate);
}

double calculateError(Network *self, double *data) {
    return self->errFunc(self, data);
}

// create & edit network
Network *createNetwork(int numStartNeurons, void (*actFunc)(struct Layer *), 
                             double (*errFunc)(Network *, double *)) {
    srand(time(0));

    Network *self = malloc(sizeof(Network));
    self->numLayers = 1;
    self->errFunc = errFunc;

    if (errFunc == MSE) {
        self->derivErrorFunc = &__MSEGradient;
    } else {
        self->derivErrorFunc = &__CEGradient;
    }

    self->layers = malloc(sizeof(*self->layers));
    struct Layer *startLayer =  &self->layers[0];
    startLayer->numNeurons = numStartNeurons;
    startLayer->neurons = malloc(numStartNeurons * sizeof(*startLayer->neurons));

    startLayer->actFunc = actFunc;

    if (actFunc == layerSigmoid) {
        startLayer->derivActFunc = &__layerSigmoidDeriv;
    } else if (actFunc == layerRelu) {
        startLayer->derivActFunc = &__layerReluDeriv;
    } else if (actFunc == layerLeakyRelu) {
        startLayer->derivActFunc = &__layerLeakyReluDeriv;
    } else if (actFunc == layerTanh) {
        startLayer->derivActFunc = &__layerTanhDeriv;
    } else if (actFunc == layerSoftmax) {
        startLayer->derivActFunc = &__layerSoftmaxDeriv;
    } else {
        startLayer->derivActFunc = &__layerLinearDeriv;
    }

    for (int i = 0; i < numStartNeurons; i++) {
        struct Neuron *neuron = &startLayer->neurons[i];
        neuron->numCons = 0;
        neuron->bias = 0;
    }

    return self;
}

void addLayer(Network *self, int numNeurons, void (*actFunc)(struct Layer *)) {
    self->numLayers++;
    self->layers = realloc(self->layers, self->numLayers * sizeof(*self->layers));

    struct Layer *newLayer = &self->layers[self->numLayers - 1];
    newLayer->numNeurons = numNeurons;
    newLayer->neurons = malloc(numNeurons * sizeof(*newLayer->neurons));
    newLayer->actFunc = actFunc;

    if (actFunc == layerSigmoid) {
        newLayer->derivActFunc = &__layerSigmoidDeriv;
    } else if (actFunc == layerRelu) {
        newLayer->derivActFunc = &__layerReluDeriv;
    } else if (actFunc == layerLeakyRelu) {
        newLayer->derivActFunc = &__layerLeakyReluDeriv;
    } else if (actFunc == layerTanh) {
        newLayer->derivActFunc = &__layerTanhDeriv;
    } else if (actFunc == layerSoftmax) {
        newLayer->derivActFunc = &__layerSoftmaxDeriv;
    } else {
        newLayer->derivActFunc = &__layerLinearDeriv;
    }

    for (int i = 0; i < numNeurons; i++) {
        struct Neuron *neuron = &newLayer->neurons[i];
        neuron->bias = __randDouble(-1.0, 1.0);
        neuron->numCons = self->layers[self->numLayers - 2].numNeurons;

        neuron->weights = malloc(neuron->numCons * sizeof(*neuron->weights));
        neuron->inputs = malloc(neuron->numCons * sizeof(*neuron->inputs));
        neuron->gradients = malloc((neuron->numCons + 1) * sizeof(*neuron->gradients));
        for (int j = 0; j < neuron->numCons; j++) {
            neuron->weights[j] = __randDouble(-1.0, 1.0);
            neuron->inputs[j] = 0;
            neuron->gradients[j] = 0;
        }
        neuron->gradients[neuron->numCons] = 0;
    }
}


//function for print outputs of last layer
void printNetworkOutput(Network *self) {
    struct Layer *layer = &self->layers[self->numLayers - 1];
    for (int i = 0; i < layer->numNeurons; i++) {
        printf("%lf\t", layer->neurons[i].output);
    }
    putchar('\n');
}

//function for get max output from last layer
int getIndexMaxOutput(Network *self) {
    struct Layer *layer = &self->layers[self->numLayers - 1];
    double max = layer->neurons[0].output;

    int index = 0;
    for (int i = 1; i < layer->numNeurons; i++) {
        if (layer->neurons[i].output > max) {
            max = layer->neurons[i].output;
            index = i;
        }
    }
    return index;
}