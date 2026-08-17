#include <stdlib.h>
# include "image.h"

Image *img_create(const unsigned char *pixels, int height, int width) {
    Image * img = malloc(sizeof(Image));
    if (img == NULL) return NULL;

    img->pixels = pixels;
    img->height = height;
    img->width = width;
    return img;
}

const unsigned char *img_row(const Image *img, int row) {
    return img->pixels + row *img->width;
}

unsigned char img_at(const Image *img, int row, int col) {
    return img->pixels + (row * img->width) + col;
}

void img_free(Image *img) {
    free(img);
}