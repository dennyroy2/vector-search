#include <assert.h>
#include <stdio.h>
#include "vectors.h"

int main(void) {
    float data[] = {1.0, 2.0, 3.0 ,4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

    int n = 3;
    int dim = 4;
    VectorStore * vs = vs_create(data, n, dim);
    assert(vs != NULL);

    assert(vs->n == n);
    assert(vs->dim == dim);

    assert(vs_get(vs , 0) == data);
    assert(vs_get(vs , 1) == data + dim);
    assert(vs_get(vs , 2) == data + 2*dim);

    assert(vs_get(vs,1)[0] == 5.0);

    vs_free(vs);

    printf("test_vectors: all passed\n");
    return 0;
}