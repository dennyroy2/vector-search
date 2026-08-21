#include "bruteforce.h"
#include "distance.h"
#include "heap.h"
#include <stdlib.h>
void bruteforce_nn(const VectorStore *vs, const float *query,int *out_id, float *out_dist) {

    if (vs->n <= 0) {
        *out_id = -1;
        return;
    }

    int n = vs->n;
    int dim = vs->dim;

    int best_id = 0;
    float best_dist = l2sq_distance(vs_get(vs, 0), query, dim);


    for (int i = 1; i < n; i++) {
        float dist = l2sq_distance(vs_get(vs, i), query, dim);
        if (dist < best_dist) {
            best_dist = dist;
            best_id = i;
        }
    }

    *out_id = best_id;
    *out_dist = best_dist;
}

int bruteforce_topk(const VectorStore * vs, const float * query, int k, int * out_ids, float * out_dists) {
    if ((k <= 0) || (vs->n <= 0)) return 0;

    MaxHeap * maxHeap = heap_create(k);
    if (maxHeap == NULL) return 0;
    Candidate out;
    for (int i = 0; i < vs->n; i++) {
        float dist = l2sq_distance(vs_get(vs, i), query, vs->dim);
        if (heap_size(maxHeap) < k) {
            heap_push(maxHeap, i, dist);
        } else {
            heap_peek(maxHeap, &out);
            if (out.dist > dist) {
                heap_pop(maxHeap, &out);
                heap_push(maxHeap, i, dist);
            }
        }
    }
    int count = heap_size(maxHeap);

    for (int j = count - 1; j >= 0; j--) {
        Candidate out;
        heap_pop(maxHeap, &out);
        *(out_dists + j) = out.dist;
        *(out_ids + j) = out.id;
    }

    heap_free(maxHeap);
    return count;
}

                