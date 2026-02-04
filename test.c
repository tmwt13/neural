#include "lib/raylib.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>     // for exit()
#include "main.c"       // assuming this contains your Mat, Network, create_network,
// feed_forward, back_prop, create_matrix, MAT_AT, etc.

#define IMG_COUNT 5
#define IMG_WIDTH 28
#define IMG_HEIGHT 28

int main(void) {
    const int WINDOW_W = 800;
    const int WINDOW_H = 1000;

    InitWindow(WINDOW_W, WINDOW_H, "NN interpolation 3 7");
    SetTargetFPS(60);

    Image imgs[IMG_COUNT];
    Image gen_imgs[IMG_COUNT];
    Texture2D textures[IMG_COUNT];
    Texture2D gen_textures[IMG_COUNT];
    Color * pixels[IMG_COUNT];

    char name_buf[100];
    int len;
    for (size_t i = 0; i < IMG_COUNT; i++) {
      len = snprintf(name_buf,100, "./assets2/%zu.png",i+1);
      printf("loading: '%s'\n", name_buf);
      imgs[i] = LoadImage(name_buf);
      if(imgs[i].data == NULL) exit(69);
      ImageResize(imgs + i, IMG_WIDTH, IMG_HEIGHT);
      textures[i] = LoadTextureFromImage(imgs[i]);
      gen_imgs[i] = GenImageColor(100, 100, BLACK);
      gen_textures[i] = LoadTextureFromImage(gen_imgs[i]);
      pixels[i] = LoadImageColors(imgs[i]);
    }

    Network nn;
    uint8_t layer_sizes[] = {3,80,24,1};
    create_network(&nn, layer_sizes, 4);
    nn.learning_rate = 0.08f;   

    Mat input     = create_matrix(3, 1);
    Mat expected  = create_matrix(1, 1);

    bool training_done = false;
    size_t epoch = 0;
    float last_loss = 0.0f;

    srand(time(NULL));
    for(size_t iters=0;!training_done && iters < 10000;iters++){
        // size_t img_id = iters % IMG_COUNT;
        size_t img_id = (float)rand() / (float)RAND_MAX * IMG_COUNT;
        if (!training_done) {
            float total_loss = 0.0f;
            size_t pixel_count = IMG_WIDTH * IMG_HEIGHT;

            for (size_t i = 0; i < pixel_count; i++) {
                int x = i % IMG_WIDTH;
                int y = i / IMG_WIDTH;

                MAT_AT(input, 0, 0) = (float)x / IMG_WIDTH;
                MAT_AT(input, 1, 0) = (float)y / IMG_HEIGHT;
                MAT_AT(input, 2, 0) = (float)(img_id) / IMG_COUNT;
                MAT_AT(expected, 0, 0) = (float)pixels[img_id][x+y*IMG_WIDTH].b / 255.0f;

                feed_forward(&nn, &input);
                float loss = back_prop(&nn, &expected,i%20==0);
                total_loss += loss;
            }

            last_loss = total_loss / (float)pixel_count;

            epoch++;
            if (epoch % 50 == 0) {
                printf("(%zu) Epoch %4zu | avg loss: %.6f\n", img_id, epoch, last_loss);
            }

        }


        // // ── DRAW ──────────────────────────────────────────────────
        // BeginDrawing();
        //     if(IsKeyPressed(KEY_E)) training_done = true;
        //     ClearBackground(RAYWHITE);

        //     DrawText("Original", 20, 20, 20, DARKGRAY);
        //     for (size_t im = 0; im < IMG_COUNT; im++) {
        //       DrawTexture(textures[im], 0, 50+50*im, WHITE);
        //     }

        //     DrawText("Neural Network reconstruction", 420, 20, 20, DARKGRAY);
        //     for (size_t im = 0; im < IMG_COUNT; im++) {
        //       DrawTexture(gen_textures[im], (uint8_t)(im/5+1)*100, 100*(1+im%5), WHITE);
        //     }

        //     DrawText(TextFormat("Epoch: %zu   Loss: %.6f", epoch, last_loss),
        //              20, WINDOW_H - 40, 20, training_done ? GREEN : ORANGE);

        //     if (training_done) {
        //         DrawText("Training finished", 420, WINDOW_H - 40, 20, GREEN);
        //     }
        // EndDrawing();
        if(last_loss < 0.01) break;
    }

    float v;
    unsigned char c;
    const Mat *out;
    for (size_t im = 0; im < IMG_COUNT; im++) {
      for (int y = 0; epoch % 100 == 0 && y < gen_imgs[im].height; y++) {
        for (int x = 0; x < gen_imgs[im].width; x++) {
          MAT_AT(input, 0, 0) = (float)x / (float)gen_imgs[im].width;
          MAT_AT(input, 1, 0) = (float)y / (float)gen_imgs[im].height;
          MAT_AT(input, 2, 0) = (float)im / IMG_COUNT;

          out = feed_forward(&nn, &input);
          v = MAT_AT(*out, 0, 0);

          c = (unsigned char)(v < 0 ? 0 : v > 1 ? 255 : v * 255.0f + 0.5f);

          ImageDrawPixel(gen_imgs + im, x, y, (Color){c, c, c, 255});
        }
      }
      UpdateTexture(gen_textures[im], gen_imgs[im].data);
        }


    Image scaledimg = ImageCopy(gen_imgs[0]);
    ImageResize(&scaledimg, 400, 400);
    Texture2D scaledtex = LoadTextureFromImage(scaledimg);
    float dz =1;
    float z = 0;
    while(!WindowShouldClose()){
      if(IsMouseButtonDown(MOUSE_LEFT_BUTTON)){
        float z= GetMouseX();
        z /= WINDOW_W;
          printf("Z: %f\n",z);

          for (int y = 0; y < scaledimg.height; y++) {
            for (int x = 0; x < scaledimg.width; x++) {
              MAT_AT(input, 0, 0) = (float)x / (float)scaledimg.width;
              MAT_AT(input, 1, 0) = (float)y / (float)scaledimg.height;
              MAT_AT(input, 2,0) = z;
              const Mat *out = feed_forward(&nn, &input);
              float v = MAT_AT(*out, 0, 0);

              unsigned char c = (unsigned char)(v < 0   ? 0 : v > 1 ? 255 : v * 255.0f + 0.5f);

              ImageDrawPixel(&scaledimg, x, y, (Color){c, c, c, 255});
            }
          }
          UpdateTexture(scaledtex, scaledimg.data);
        }
        BeginDrawing();
            ClearBackground(RAYWHITE);

            DrawText("Original", 20, 20, 20, DARKGRAY);
            for (size_t im = 0; im < IMG_COUNT; im++) {
              DrawTexture(textures[im], 0, 50+50*im, WHITE);
            }

            DrawText("Neural Network reconstruction", 420, 20, 20, DARKGRAY);
            for (size_t im = 0; im < IMG_COUNT; im++) {
              DrawTexture(gen_textures[im], (uint8_t)(im/5+1)*100, 100*(1+im%5), WHITE);
            }
            DrawTexture(scaledtex, 100, 600, WHITE);

            DrawText(TextFormat("Epoch: %zu   Loss: %.6f", epoch, last_loss),
                     20, WINDOW_H - 40, 20, training_done ? GREEN : ORANGE);

            if (training_done) {
                DrawText("Training finished", 420, WINDOW_H - 40, 20, GREEN);
            }
        EndDrawing();
    }

    // Cleanup
    for (size_t i = 0; i < IMG_COUNT; i++) {
      UnloadImage(imgs[i]);
      UnloadImage(gen_imgs[i]);
      UnloadTexture(textures[i]);
      UnloadTexture(gen_textures[i]);
      free(pixels[i]);
    }
    UnloadImage(scaledimg);
    UnloadTexture(scaledtex);

    // TODO: free neural network memory if your implementation needs it

    CloseWindow();
    return 0;
}
