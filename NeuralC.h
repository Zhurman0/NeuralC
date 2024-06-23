#ifndef NEURALC_H
#define NEURALC_H

struct Layer;
typedef struct Network Network;

void layerLinear(struct Layer *layer);
void layerRelu(struct Layer *layer);
void layerLeakyRelu(struct Layer *layer);
void layerSigmoid(struct Layer *layer);
void layerTanh(struct Layer *layer);
void layerSoftmax(struct Layer *layer);

double MSE(Network *self, double *targets);
double CE(Network *self, double *targets);

void forwardPass(Network *self, double *data);
void backPropagation(Network *self, double *target, double learningRate);
double calculateError(Network *self, double *data);

Network *createNetwork(int numStartNeurons, void (*actFunc)(struct Layer *), 
                             double (*errFunc)(Network *, double *));
void addLayer(Network *self, int numNeurons, void (*actFunc)(struct Layer *));

void printNetworkOutput(Network *self);
int getIndexMaxOutput(Network *self);

#endif