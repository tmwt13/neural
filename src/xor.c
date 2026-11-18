#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define NN_IMPLEMENTATION
#include "neural.h"
#include <time.h>

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
    loss = back_prop(&nn, &expeced_out, true);
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

