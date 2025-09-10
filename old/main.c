// gcc -o main main.c $(pkg-config --libs --cflags raylib)
// NN library
#define NN_IMPLEMENTATION
#include "./mylib.h"

// other stuff
#include "./raylib.h"
#include <stddef.h>

void draw_on_screen();
void draw_matrix(Mat, int, int);

int main(void) {

  // Initialize the window
  InitWindow(800, 800, "Hello Welt");

  Mat a = create_matrix(3, 3);
  mat_random(&a);

  // ende
  while (!WindowShouldClose()) {
    draw_on_screen();
  }
  CloseWindow();
  return 0;
}

void draw_on_screen() {
  BeginDrawing();
  ClearBackground(LIGHTGRAY);
  DrawText("Hallo Welt", 0, 0, 20, BLUE);

  Mat a = create_matrix(3, 5);
  mat_random(&a);

  draw_matrix(a, 200, 200);

  EndDrawing();

  WaitTime(0.6);
}

void draw_matrix(Mat a, int x, int y) {
  char str[100];
  for (size_t r = 0; r < a.rows; r++) {
    for (size_t c = 0; c < a.cols; c++) {
      sprintf(str, "%f", MAT_AT(a, r, c));
      DrawText(str, x + c * 60, y + r * 20, 10, BLACK);
    }
  }
}
