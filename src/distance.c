#include "distance.h"

float l2sq_distance(const float * restrict a, const float * restrict b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = (a[i] - b[i]);
        sum += diff * diff;
    }
    return sum;
}