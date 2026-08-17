#ifndef VECTORS_H
#define VECTORS_H
typedef struct {
    const float *data;   
    int n;               
    int dim;            
} VectorStore;


VectorStore *vs_create(const float *data, int n, int dim);

const float *vs_get(const VectorStore *vs, int i);
void vs_free(VectorStore *vs);

#endif