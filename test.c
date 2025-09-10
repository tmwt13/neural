#define NN_IMPLEMENTATION
#include "matrix.h"

int main(void) {

  Mat a;
  Mat b;
  a = create_matrix(3, 3);
  b = create_matrix(3, 2);
  mat_random(&b);
  mat_random(&a);
  mat_print(a);
  Mat c = mat_mult(a, b);
  mat_print(c);
  return 0;
}
