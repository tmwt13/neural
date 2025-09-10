// #include <Foundation/Foundation.h>
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

#define ASSERT(exp, msg) assert((exp) && (msg))
#define MAT_OFFSET_AT(mat, r, c) (r) * (mat).cols + (c)
#define MAT_AT(mat, r, c) *((mat).data + MAT_OFFSET_AT(mat, r, c))

typedef struct {
  size_t rows, cols;
  double *data;
} Mat;

Mat create_matrix(size_t rows, size_t cols);
void mat_fill(Mat *a, double val);
void mat_random(Mat *a);
Mat mat_transpose(Mat a);
void mat_print(Mat a);
void mat_add(Mat a, Mat b);
void mat_sub(Mat a, Mat b);
Mat mat_mult(Mat a, Mat b);

// #define NN_IMPLEMENTATION
#ifdef NN_IMPLEMENTATION
// mylib.c
double AddTwo(double a, double b) {
  double aa = a * 2;
  double bb = b * 3;
  return aa + bb;
}

const char *print_to_console(const char *msg) {
  // for now just return the input back again
  return msg;
}

Mat create_matrix(size_t rows, size_t cols) {
  double *data = malloc(sizeof(double) * rows * cols);
  return (Mat){
      rows = rows,
      cols = cols,
      data = data,
  };
}

void mat_fill(Mat *a, double val) {
  for (size_t r = 0; r < a->rows; r++) {
    for (size_t c = 0; c < a->cols; c++) {
      MAT_AT(*a, r, c) = val;
    }
  }
}

void mat_random(Mat *a) {
  for (size_t r = 0; r < a->rows; r++) {
    for (size_t c = 0; c < a->cols; c++) {
      MAT_AT(*a, r, c) = (rand() / (double)RAND_MAX) * 2 - 1;
    }
  }
}

Mat mat_transpose(Mat a) {
  Mat t = create_matrix(a.cols, a.rows);

  for (size_t r = 0; r < a.rows; r++) {
    for (size_t c = 0; c < a.cols; c++) {
      MAT_AT(t, c, r) = MAT_AT(a, r, c);
    }
  }

  return t;
}

void mat_print(Mat a) {
  printf("\n");
  for (size_t r = 0; r < a.rows; r++) {
    for (size_t c = 0; c < a.cols; c++) {
      printf("%f   ", MAT_AT(a, r, c));
    }
    printf("\n");
  }
}

void mat_add(Mat a, Mat b) {
  // TEST for equality
  ASSERT(a.cols == b.cols && a.rows == b.rows, "rows and cols are not equal!");
  for (size_t r = 0; r < a.rows; r++) {
    for (size_t c = 0; c < b.cols; c++) {
      MAT_AT(a, c, r) += MAT_AT(b, r, c);
    }
  }
}

void mat_sub(Mat a, Mat b) {
  // TEST for equality
  ASSERT(a.cols == b.rows, "rows and cols are not equal!");
  for (size_t r = 0; r < a.rows; r++) {
    for (size_t c = 0; c < b.cols; c++) {
      MAT_AT(a, c, r) -= MAT_AT(b, r, c);
    }
  }
}

Mat mat_mult(Mat a, Mat b) {
  ASSERT(a.cols == b.rows, "ROWS and COLS must match!");
  Mat result = create_matrix(a.rows, b.cols);
  for (size_t r = 0; r < a.rows; r++) {
    for (size_t c = 0; c < b.cols; c++) {
      for (size_t k = 0; k < a.cols; k++) {
        MAT_AT(result, r, c) += MAT_AT(a, r, k) * MAT_AT(b, c, k);
      }
    }
  }

  return result;
}

int test() {
  Mat a;
  Mat b;
  Mat c;

  a = create_matrix(3, 3);
  mat_random(&a);
  mat_print(a);

  Mat aa = mat_mult(a, a);
  mat_print(aa);

  return 0;
}
#endif
