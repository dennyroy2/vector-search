#include <stdlib.h>
#include "vectors.h"

VectorStore * vs_create(const float * data, int n, int dim) {
    VectorStore * vector = malloc(sizeof(VectorStore));
    if (vector == NULL) return NULL;

    vector->data = data;
    vector->n = n;
    vector->dim = dim;

    return vector;
}

const float *vs_get(const VectorStore *vs, int i) {
    return vs->data + i*vs->dim;
}

void vs_free(VectorStore *vs) {
    free(vs);
}
