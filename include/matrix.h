// #include <Foundation/Foundation.h>
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <stdbool.h>

#define ASSERT(exp, msg) assert((exp) && (msg))
#define MAT_OFFSET_AT(mat, r, c) (r) * (mat).cols + (c)
#define MAT_AT(mat, r, c) *((mat).data + MAT_OFFSET_AT(mat, r, c))
#define MAT_SAME_DIM(a,b) ((a)->rows == (b)->rows && (a)->cols == (b)->cols)

typedef struct {
  size_t rows, cols;
  float *data;
} Mat;

Mat create_matrix(size_t rows, size_t cols);
void mat_free(Mat *a);
void mat_fill(Mat *a, float val);
void mat_copy(Mat *dest, const Mat* src);
void mat_random(Mat *a);
void mat_transpose(Mat *dest, Mat *src);
void mat_print(Mat a);
void mat_add(Mat *a, Mat *b);
void mat_sub(Mat *a,const Mat *b);
void mat_sub_with_factor(Mat *dest, const Mat *sub, float factor);
void mat_mult(Mat *res, Mat *a, Mat *b, bool clear);
void mat_mult_nn(Mat *res, Mat *a, Mat *b, bool clear);
void mat_mult_nt(Mat *res, Mat *a, Mat *b, bool clear);
void mat_mult_tn(Mat *res, Mat *a, Mat *b, bool clear);
void mat_mult_tt(Mat *res, Mat *a, Mat *b, bool clear);
void mat_scalar_mult(Mat *b, Mat *a, float s);

#ifndef MAT_IMPLEMENTATION
#define MAT_IMPLEMENTATION

const char *print_to_console(const char *msg) {
  // for now just return the input back again
  return msg;
}

Mat create_matrix(size_t rows, size_t cols) {
  float *data = malloc(sizeof(float) * rows * cols);
  return (Mat){
      .rows = rows,
      .cols = cols,
      .data = data,
  };
}

void mat_free(Mat *a){
  free(a->data);
  a->rows=0;
  a->cols=0;
}

void mat_fill(Mat *a, float val) {
  for (size_t r = 0; r < a->rows; r++) {
    for (size_t c = 0; c < a->cols; c++) {
      MAT_AT(*a, r, c) = val;
    }
  }
}

void mat_random(Mat *a) {
  for (size_t r = 0; r < a->rows; r++) {
    for (size_t c = 0; c < a->cols; c++) {
      MAT_AT(*a, r, c) = (rand() / (float)RAND_MAX) * 2 - 1;
    }
  }
}

#define PNAME(v) #v
void mat_transpose(Mat *dest, Mat *source) {
  ASSERT(dest->cols==source->rows&&dest->rows==source->cols, "ERROR");
  for (size_t r = 0; r < source->rows; r++) {
    for (size_t c = 0; c < source->cols; c++) {
      MAT_AT(*dest, c, r) = MAT_AT(*source, r, c);
    }
  }
}

void mat_print(Mat a) {
  for (size_t r = 0; r < a.rows; r++) {
    for (size_t c = 0; c < a.cols; c++) {
      printf("%f   ", MAT_AT(a, r, c));
    }
    printf("\n");
  }
}

void mat_add(Mat *a, Mat *b) {
  ASSERT(MAT_SAME_DIM(a,b), "rows and cols are not equal!");
  for (size_t i = 0; i < a->rows*a->cols; i++) {
    a->data[i] += b->data[i];
  }
}

void mat_sub(Mat *a,const Mat *b) {
  ASSERT(MAT_SAME_DIM(a,b), "rows and cols are not equal!");
  for (size_t i = 0; i < a->rows*a->cols; i++) {
    a->data[i] -= b->data[i];
  }
}

void mat_sub_with_factor(Mat *dest, const Mat *sub, float factor) {
  ASSERT(MAT_SAME_DIM(dest, sub), "rows and cols are not equal!");
  for (size_t i = 0; i < dest->rows*dest->cols; i++) {
    dest->data[i] -= sub->data[i] * factor;
  }
}

void mat_mult(Mat *res, Mat *a, Mat *b, bool clear) {
  mat_mult_nn(res,a,b,clear);
}

void mat_scalar_mult(Mat *b, Mat *a, float s){
  //printf("(%zu,%zu) -> (%zu,%zu)",a->rows,a->cols,b->rows,b->cols);
  ASSERT(MAT_SAME_DIM(a, b), "ERROR");
  for(size_t i =0;i<a->cols*a->rows;i++){
    b->data[i] = a->data[i] * s;
  }
}

void mat_mult_nn(Mat *res, Mat *a, Mat *b, bool clear){
  ASSERT(a->cols == b->rows, "ROWS and COLS must match!");
  ASSERT(res->rows == a->rows, "ROWS and COLS must match!");
  ASSERT(res->cols == b->cols, "ROWS and COLS must match!");
  if (clear) mat_fill(res, 0);
  for (size_t r = 0; r < a->rows; r++) {
    for (size_t k = 0; k < a->cols; k++) {
      for (size_t c = 0; c < b->cols; c++) {
        res->data[r*res->cols + c] += a->data[r*a->cols+k] * b->data[k*b->cols+c];
      }
    }
  }
}
void mat_mult_nt(Mat *res, Mat *a, Mat *b, bool clear) {
  ASSERT(a->cols == b->cols, "Inner dimensions must match (a.cols == b.cols)");
  if (clear)
    mat_fill(res, 0);
  for (size_t r = 0; r < a->rows; r++) {
    for (size_t c = 0; c < b->rows; c++) {
      for (size_t k = 0; k < a->cols; k++) {
        // b ist transponiert -> Zugriff auf Zeile c, Spalte k
        res->data[r * res->cols + c] +=
            a->data[r * a->cols + k] * b->data[c * b->cols + k];
      }
    }
  }
}
void mat_mult_tn(Mat *res, Mat *a, Mat *b, bool clear){
  ASSERT(a->rows == b->rows, "ROWS and COLS must match!");
  ASSERT(res->rows == a->cols, "ROWS and COLS must match!");
  ASSERT(res->cols == b->cols, "ROWS and COLS must match!");
  if (clear) mat_fill(res, 0);
  for (size_t k = 0; k < a->rows; k++) {
    for (size_t r = 0; r < a->cols; r++) {
      for (size_t c = 0; c < b->cols; c++) {
        res->data[r*res->cols + c] += a->data[k*a->cols+r] * b->data[k*b->cols+c];
      }
    }
  }
}

void mat_mult_tt(Mat *res, Mat *a, Mat *b, bool clear){
  printf("%zux%zu -> %zux%zu\n",  a->rows, a->cols,b->rows, b->cols);
  ASSERT(a->rows == b->cols, "ROWS and COLS must match!");
  ASSERT(res->rows == a->cols, "ROWS and COLS must match!");
  ASSERT(res->cols == b->rows, "ROWS and COLS must match!");
  if (clear) mat_fill(res, 0);
  for (size_t r = 0; r < a->cols; r++) {
    for (size_t c = 0; c < b->cols; c++) {
      for (size_t k = 0; k < a->rows; k++) {
        res->data[r*res->cols + c] += a->data[k*a->cols+r] * b->data[c*b->cols+k];
      }
    }
  }
}

void mat_copy(Mat *dest, const Mat* src){
  // printf("%zux%zu -> %zux%zu\n",  src->rows, src->cols,dest->rows, dest->cols);
  ASSERT(MAT_SAME_DIM(dest, src), "ERROR");
  // printf("%p -> %p\n", src->data, dest->data);
  // TODO: WARUM??
  memcpy(dest->data, src->data, src->cols*src->rows*sizeof(float));
}
#endif
