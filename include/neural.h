#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include <time.h>

typedef struct {
  Mat input, out_grad, weights, bias, dweights, dbias, output, in_grad;  
}FCL;

typedef struct {
  float *data;
  size_t data_size, layer_count;
  FCL *layer;
  float learning_rate, loss;
} Network;

void random_fill(float* data, size_t len);

void create_network(Network *nn, const uint8_t *layer, const uint8_t len);
void init_threads(Network* nn);

const Mat* feed_forward(Network* nn,const Mat* input);
void layer_feed_forward(FCL* layer);

float back_prop(Network* nn, const Mat* expected_out, bool update);
void layer_backward(FCL* layer, float learing_rate,bool update);
void mse_loss_and_grad(Mat *grad, const Mat *pred, const Mat *target);
void sigmoid(Mat *dest, const Mat *src);
void sigmoid_deriv(Mat *dest, const Mat *output);
void activ(Mat *dest, Mat *src);
void activ_grad(Mat *dest, Mat *output);
void relu(Mat *dest, Mat *src);
void relu_grad(Mat *dest, Mat *output);

#ifdef NN_IMPLEMENTATION

void sigmoid(Mat *dest, const Mat *src) {
    for (size_t i = 0; i < src->rows * src->cols; ++i) {
        dest->data[i] = 1.0f / (1.0f + expf(-src->data[i]));
    }
}

void sigmoid_deriv(Mat *dest, const Mat *output) {
    for (size_t i = 0; i < output->rows * output->cols; ++i) {
        float y = output->data[i];
        dest->data[i] *= y * (1.0f - y);
    }
}

void mse_loss_and_grad(Mat *grad, const Mat *pred, const Mat *target) {
    float sum = 0.0f;
    for (size_t i = 0; i < pred->rows; ++i) {
        float err = MAT_AT(*pred, i, 0) - MAT_AT(*target, i, 0);
        MAT_AT(*grad, i, 0) = 2.0f * err;  // derivative of MSE w.r.t. output
        sum += err * err;
    }
    // return sum / pred->rows;  // if you want actual mean
}

static int ctoi(const char* text){
  int res;
  for(uint8_t i=0;i<10 && text[i] != 0;i++){
    if(text[i] < 48 || text[i] > 57) return -1;
    res = res*10 + text[i]-48;
  }
  return res;
}

void create_network(Network *nn, const uint8_t *layer, const uint8_t len){
  #ifdef NN_MULTITHREAD
    printf(" -- MULTITHREADING: ON -- \n");
    init_threads(nn);
  #endif
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
  }
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


void layer_feed_forward(FCL* layer){
  mat_mult(&layer->output, &layer->weights, &layer->input,1);
  mat_add(&layer->output, &layer->bias);

  sigmoid(&layer->output, &layer->output);
}

void layer_backward(FCL* layer, float learing_rate,bool update){
  sigmoid_deriv(&layer->in_grad, &layer->output);
 
  // update out grads
  mat_mult_tn(&layer->out_grad, &layer->weights, &layer->in_grad, 1);
  // continue with next layer on next thread -> 
  // next_thread_calc_grad();

  // do simultanious
  mat_mult_nt(&layer->dweights, &layer->in_grad, &layer->input, 0);
  mat_add(&layer->dbias, &layer->in_grad);

  // update weights
  if (update) {
    mat_sub_with_factor(&layer->weights, &layer->dweights, learing_rate);
    mat_sub_with_factor(&layer->bias, &layer->dbias, learing_rate);

    mat_fill(&layer->dweights, 0);
    mat_fill(&layer->dbias, 0);
  }
}

const Mat* feed_forward(Network* nn, const Mat* input) {
    mat_copy(&nn->layer[0].input, input);

    for (size_t i = 0; i < nn->layer_count; ++i) {
        layer_feed_forward(&nn->layer[i]);
    }

    return &nn->layer[nn->layer_count-1].output;
}

#include <pthread.h>
#include <semaphore.h>
#define THREAD_COUNT 2
pthread_t threads[THREAD_COUNT];
int ids[THREAD_COUNT];
typedef struct{
  int *data;
  size_t count;
  size_t capacity;
  pthread_mutex_t mutex;
  pthread_cond_t cond_done;
  pthread_cond_t cond_work;
  size_t layers_done;
  bool update;
  Network *network;
} Layers;

#define DA_APPEND(ptr, item)                                                   \
  do {                                                                         \
    if ((ptr)->count == (ptr)->capacity) {                                     \
      if ((ptr)->capacity == 0)                                                \
        (ptr)->capacity = 1;                                                   \
      (ptr)->capacity *= 2;                                                    \
      (ptr)->data = realloc((ptr)->data, sizeof((ptr)->data[0]) * (ptr)->capacity * 2);  \
      if ((ptr)->data == NULL)                                                 \
        exit(69);                                                              \
    }                                                                          \
    (ptr)->data[(ptr)->count++] = item;                                        \
  } while (0)

#define DA_POP(ptr) (ptr)->data[((ptr)->count--) - 1]

Layers proc_list = {0};

void process_layer(int layer_id){
    if(layer_id < 0) return;
    pthread_mutex_lock(&proc_list.mutex);
    DA_APPEND(&proc_list, layer_id);
    pthread_cond_broadcast(&proc_list.cond_work);
    pthread_mutex_unlock(&proc_list.mutex);
}

void layer_backward_mt(Network *nn, int layer_id, bool update){
  if(layer_id < 0) return;
  FCL* layer = &nn->layer[layer_id];
  sigmoid_deriv(&layer->in_grad, &layer->output);
 
  // update out grads
  mat_mult_tn(&layer->out_grad, &layer->weights, &layer->in_grad, 1);
  // continue with next layer on next thread -> 
  process_layer(layer_id-1);
  
  // do simultanious
  mat_mult_nt(&layer->dweights, &layer->in_grad, &layer->input, 0);
  mat_add(&layer->dbias, &layer->in_grad);

  // update weights
  if (update) {
    mat_sub_with_factor(&layer->weights, &layer->dweights, nn->learning_rate);
    mat_sub_with_factor(&layer->bias, &layer->dbias, nn->learning_rate);

    mat_fill(&layer->dweights, 0);
    mat_fill(&layer->dbias, 0);
  }

  pthread_mutex_lock(&proc_list.mutex);
  proc_list.layers_done++;
  pthread_mutex_unlock(&proc_list.mutex);
}
#include <string.h>
#include <errno.h>
#include <fcntl.h>

void *work(void *args) {
  // Layers *proc_list = (Layers*)da_layers;
  int id = *(int*)args;
  bool update;
  while (1) {
    pthread_mutex_lock(&proc_list.mutex);
    if (proc_list.count == 0) {
      if (proc_list.layers_done == proc_list.network->layer_count) pthread_cond_broadcast(&proc_list.cond_done);
      pthread_cond_wait(&proc_list.cond_work, &proc_list.mutex);
      update = proc_list.update;
      pthread_mutex_unlock(&proc_list.mutex);
      continue;
    }
    int layer_id = DA_POP(&proc_list);
    pthread_mutex_unlock(&proc_list.mutex);
    layer_backward_mt(proc_list.network, layer_id, update);
  }
}

int ids[THREAD_COUNT];
void init_threads(Network* nn) {
  proc_list = (Layers){0};
  pthread_mutex_init(&proc_list.mutex, NULL);
  pthread_cond_init(&proc_list.cond_done, NULL);
  pthread_cond_init(&proc_list.cond_work, NULL);
  proc_list.network = nn;
  
  for(size_t i=0;i<THREAD_COUNT;i++){
    ids[i] = i;
    pthread_create(threads+i, NULL, work, ids+i);
  }
}

float back_prop(Network *nn, const Mat *target, bool update) {
  Mat output_grad = nn->layer[nn->layer_count - 1].in_grad;

  mse_loss_and_grad(&output_grad, &nn->layer[nn->layer_count - 1].output, target);
  
#ifdef NN_MULTITHREAD
  pthread_mutex_lock(&proc_list.mutex);
  proc_list.update = update;
  proc_list.layers_done = 0;
  pthread_mutex_unlock(&proc_list.mutex);

  layer_backward_mt(nn, nn->layer_count - 1, update);
  pthread_mutex_lock(&proc_list.mutex);
  if (proc_list.layers_done != nn->layer_count) pthread_cond_wait(&proc_list.cond_done, &proc_list.mutex);
  proc_list.layers_done = 0;
  pthread_mutex_unlock(&proc_list.mutex);
#else
  for (int i = (int)nn->layer_count - 1; i >= 0; --i) {
    layer_backward(nn->layer + i, nn->learning_rate, update);
  }
#endif

    float loss = 0.0f;
    Mat *out = &nn->layer[nn->layer_count-1].output;
    for (size_t i = 0; i < out->rows; ++i) {
        float e = MAT_AT(*out, i, 0) - MAT_AT(*target, i, 0);
        loss += e * e;
    }
    return loss;
}

void random_fill(float* data, size_t len){
  srand(time(NULL));
  for (size_t i=0;i<len;i++){
    data[i] = (float)rand()/(float)RAND_MAX * 2.0f - 1;
  
  }
}
#endif
