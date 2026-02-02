#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define NN_IMPLEMENTATION
#include "matrix.h"
#include <time.h>

typedef struct{
  Mat input, out_grad, weights, bias, dweights, dbias, output, in_grad;  
} FCL;

typedef struct{
  float *data;
  size_t data_size, layer_count;
  FCL *layer;
  float learning_rate, loss;
} Network;

void random_fill(float* data, size_t len);
int ctoi(const char* text){
  int res;
  for(uint8_t i=0;i<10 && text[i] != 0;i++){
    if(text[i] < 48 || text[i] > 57) return -1;
    res = res*10 + text[i]-48;
  }
  return res;
}

void create_network(Network *nn, const uint8_t *layer, const uint8_t len){
  size_t data_size = 0;
  data_size += layer[0]; // inputs first layer
  data_size += layer[0]; // out_grad first layer
  for(uint8_t i=0;i<len-1;i++){
    size_t insize = layer[i];
    size_t outsize = layer[i+1];
    data_size += insize * outsize; // weights
    data_size += outsize; // bias
    data_size += insize * outsize; // weight gradients
    data_size += outsize; // bias gradients
    data_size += outsize; // output
    data_size += outsize; // grads prev layer
  }
  
  // alloc data area
  nn->data = malloc(sizeof(float) * data_size);
  if (nn->data == NULL) exit(69);
  random_fill(nn->data, data_size);
  nn->data_size = data_size;
  nn->layer_count = len-1;
  nn->layer = malloc(sizeof(FCL)*nn->layer_count);
    if (nn->layer == NULL) exit(69);

  //create layers
  float *ptr = nn->data;
  for(uint8_t i=0;i<len-1;i++){
    size_t insize = layer[i];
    size_t outsize = layer[i+1];
    printf("\tLayer %u: %zu -> %zu\n",i,insize,outsize);
    FCL *fcl = nn->layer+i;
    printf("INPUT AT %p\n",ptr);
    fcl->input = (Mat){
      .rows = insize,
      .cols = 1,
      .data = ptr,
    };
    ptr += insize;
    fcl->out_grad = (Mat){
      .rows = insize,
      .cols = 1,
      .data = ptr,
    };
    ptr+=insize;

    fcl->weights = (Mat){
      .rows = outsize,
      .cols = insize,
      .data = ptr,
    };
    ptr += insize * outsize; // weights

    fcl->bias = (Mat){
      .rows = outsize,
      .cols = 1,
      .data = ptr,
    };
    ptr += outsize; // bias

    fcl->dweights = (Mat){
      .rows = outsize,
      .cols = insize,
      .data = ptr,
    };
    ptr += insize * outsize; // weight gradients

    fcl->dbias = (Mat){
      .rows = outsize,
      .cols = 1,
      .data = ptr,
    };
    ptr += outsize; // bias gradients

    fcl->output = (Mat){
      .rows = outsize,
      .cols = 1,
      .data = ptr,
    };
    fcl->in_grad = (Mat){
      .rows = outsize,
      .cols = 1,
      .data = ptr+outsize,
    };
    printf("OUTPUT AT %p\n",ptr);
    // da gleichzeitig input von nächstem ptr += outsize; // output / grads prev layer
  }
}

void layer_feed_forward(FCL* layer);
void layer_backward_prop(FCL* layer, float learing_rate);
const Mat* feed_forward(Network* nn,const Mat* input);
float back_prop(Network* nn, const Mat* expected_out);

float loss(const Mat *result,const Mat *expected){
  ASSERT(result->rows == expected->rows, "ERROR");
  float l = 0;
  for (uint8_t r = 0; r < result->rows; r++) {
    l+=(MAT_AT(*result,r,0)-MAT_AT(*expected, r,0));
  }
  return 2.0f * l;
}

void activ(Mat *dest, Mat *src){
  ASSERT(MAT_SAME_DIM(src,dest),"ERRROR");
  for(size_t i=0;i<dest->rows*dest->cols;i++){
    dest->data[i] = tanhf(src->data[i]);
  }
}

void activ_grad(Mat *dest, Mat *output){
  ASSERT(MAT_SAME_DIM(output,dest),"ERRROR");
  for(size_t i=0;i<dest->rows*dest->cols;i++){
    float y = output->data[i]; 
    dest->data[i] *= (1.0f - y * y);
  }
}

void relu(Mat *dest, Mat *src){
  ASSERT(MAT_SAME_DIM(src,dest),"ERRROR");
  for(size_t i=0;i<dest->rows*dest->cols;i++){
    if (src->data[i] > 0) {
      dest->data[i] = src->data[i];
      continue;
    }
    dest->data[i] = 0;
  }
}

void relu_grad(Mat *dest, Mat *output) {
  // TODO: könnte auch einfach der vorzeichen bit sein.
  ASSERT(MAT_SAME_DIM(output, dest), "ERRROR");
  for (size_t i = 0; i < dest->rows * dest->cols; i++) {
    if (output->data[i] > 0) {
      // dest->data[i] *= 1;
      continue;
    }
    dest->data[i] = 0;
  }
}

int main(){
  printf("Hallo Welt!");

  // XOR:
  //   0,0 -> 0
  //   1,0 -> 1
  //   0,1 -> 1
  //   1,1 -> 0
  // Layer:
  //   2x4 -> activ -> 4x1 -> active -> out

  float data[4][3] = {
    {0,0,0},
    {1,0,1},
    {0,1,1},
    {1,1,0},
  };

  Network nn;
  uint8_t layer[3] = {2,4,1};
  create_network(&nn, layer, 3);
  nn.learning_rate = 0.1;

  Mat input = create_matrix(2, 1);
  Mat expeced_out = create_matrix(1, 1);
  const Mat *out;
  size_t data_idx = 0;
  float loss;
  nn.learning_rate = 1;
  for (size_t iters = 0; iters < 600; iters++) {
    data_idx = iters % 4;
    nn.learning_rate *= 0.999;
    MAT_AT(input, 0, 0) = data[data_idx][0];
    MAT_AT(input, 1, 0) = data[data_idx][1];
    MAT_AT(expeced_out, 0, 0) = data[data_idx][2];
    out = feed_forward(&nn, &input);
    loss = back_prop(&nn, &expeced_out);
    if (iters % 20== 0) printf("Pass: %zu, Loss: %f\n", iters, loss*loss);
  }

  char buf[20];
  for (;;) {
    printf("Input A: ");
    if (scanf("%19s", buf) != 1) break; // Liest String ein
    MAT_AT(input, 0, 0) = (float)atof(buf); // atof ist Standard für "ascii to float"

    printf("Input B: ");
    if (scanf("%19s", buf) != 1) break;
    MAT_AT(input, 1, 0) = (float)atof(buf);

    out = feed_forward(&nn, &input);
    printf("Result: ");
    mat_print(*out);
}
  return 0;
}

void layer_feed_forward(FCL* layer){
  mat_mult(&layer->output, &layer->weights, &layer->input,1);
  mat_add(&layer->output, &layer->bias);

  activ(&layer->output, &layer->output);
}

void layer_backward_prop(FCL* layer, float learing_rate){
  activ_grad(&layer->in_grad, &layer->output);
 
  mat_mult_nt(&layer->dweights, &layer->in_grad, &layer->input, 1);
  // mat_copy(&layer->dbias, &layer->in_grad);
  mat_mult_tn(&layer->out_grad, &layer->weights, &layer->in_grad, 1);

  // update weights
  mat_sub_with_factor(&layer->weights, &layer->dweights, learing_rate);
  mat_sub_with_factor(&layer->dbias, &layer->in_grad, learing_rate);
}

const Mat* feed_forward(Network* nn,const Mat* input){
  Mat* first_input = &nn->layer[0].input;
  mat_copy(first_input, input);
  for(uint8_t i=0;i<nn->layer_count;i++){
    layer_feed_forward(nn->layer + i);
  }

  return &nn->layer[nn->layer_count-1].output;
}

float back_prop(Network* nn, const Mat* expected_out){
  Mat* output = &nn->layer[nn->layer_count-1].output;
  Mat* first_grad = &nn->layer[nn->layer_count-1].in_grad;
  nn->loss = loss(output, expected_out);

  // loss grad:
  mat_copy(first_grad, output);
  mat_sub(first_grad, expected_out);
  
  for (int i = nn->layer_count - 1; i >= 0; i--) {
    layer_backward_prop(nn->layer+i, nn->learning_rate);
  }

  return nn->loss;
}

void random_fill(float* data, size_t len){
  srand(time(NULL));
  for (size_t i=0;i<len;i++){
    data[i] = (float)rand()/(float)RAND_MAX * 2.0f - 1;
  }
}
