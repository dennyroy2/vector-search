#ifndef IMAGE_H
#define IMAGE_H

typedef struct {
    const unsigned char * pixels;
    int height;
    int width;
} Image;

Image * img_create(const unsigned char * pixels, int height, int width);
const unsigned char * img_row(const Image * img, int row);
unsigned char img_at(const Image * img, int row, int col);
void img_free(Image * img);

#endif