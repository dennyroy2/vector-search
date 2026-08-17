#include "vecops.h"

void scale_array(float * data, int n, float factor) {
    for (int i = 0; i < n; i++) {
        data[i] = data[i] * factor;
    }
}