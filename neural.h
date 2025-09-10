#define NN_IMPLEMENTATION
#include "matrix.h"

typedef struct {

} Network;

void add_fcl(Network *net, size_t input, size_t output);

// forward pass
void forward(Network *net);
// backprob
double loss(Mat *output);
void grads(Network *net, double loss);
void optimise(Network *net, double lr);

#define NEURAL_IMPLEMENTATION // to delete
#ifdef NEURAL_IMPLEMENTATION

#endif
