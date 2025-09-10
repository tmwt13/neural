#define NN_IMPLEMENTATION
#include "matrix.h"

int main(void) {

  Mat x; // inputs 3x1
  Mat w; // weights 2x3
  Mat b; // bias 2x1

  Mat o; // outputs 2x1

  x = create_matrix(3, 1);
  mat_random(&x);
  w = create_matrix(2, 3);
  mat_random(&w);
  b = create_matrix(2, 1);
  mat_random(&b);

  o = mat_mult(w, x);
  mat_add(o, b);
  mat_print(o);

  return 0;
}
