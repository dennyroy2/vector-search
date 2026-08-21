#ifndef BRUTEFORCE_H
#define BRUTEFORCE_H
#include "vectors.h"

void bruteforce_nn(const VectorStore * vs, const  float * query, int * out_id, float * out_dist);

int bruteforce_topk(const VectorStore * vs, const float * query, int k, int * out_ids, float * out_dists);
#endif