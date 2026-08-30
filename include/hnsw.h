#ifndef HNSW_H
#define HNSW_H
#define HNSW_MAX_LEVEL 16
#define HNSW_MAGIC   0x484E5357   /* "HNSW" */
#define HNSW_VERSION 1

#include "graph.h"
#include "vectors.h"
#include "heap.h"
#include "distance.h"
#include <stddef.h>

// mL : decay parameter. The paper recommends 1/ln(M), which makes each
//      layer hold roughly 1/M of the layer below and minimises overlap
//      between consecutive layers.
//
// Returns a value in [0, HNSW_MAX_LEVEL]. Most calls return 0.
int hnsw_random_level(double mL);

static inline double hnsw_default_mL(int M);

typedef struct {
    const VectorStore * vs;
    Graph ** layers;
    int * node_levels;
    int max_level;
    int entry_point;
    int n;
    int M;
    int M0;
    double mL;
} HNSW;

HNSW * hnsw_create(VectorStore * vs, int M);

void hnsw_free(HNSW * h);

int hnsw_ensure_level(HNSW * h, int target);

int hnsw_node_in_layer(const HNSW * h, int node, int layer);

Graph * hnsw_layer(HNSW * h, int layer);

int hnsw_insert(HNSW *h, int node, int ef_construction,
                VisitedSet *visited, MaxHeap *candidates, MaxHeap *results,
                int *found_ids, float *found_dists, int *selected);

// Build the whole index by inserting every vector.
int hnsw_build(HNSW *h, int ef_construction, int seed);


// Search the index for the k nearest neighbours of `query`.
//
// ef_search : beam width at layer 0. Must be >= k. The tuning knob.
//
// Returns the number written, ascending by distance.
int hnsw_search(HNSW *h, const float *query, int k, int ef_search,
                VisitedSet *visited, MaxHeap *candidates, MaxHeap *results,
                int *out_ids, float *out_dists, int *out_ndists, int * out_descent_ndists);

// Write the index to disk. Saves the graph structure only — the vectors
// belong to the caller and are loaded separately.
// Returns 1 on success, 0 on failure.
int hnsw_save(HNSW *h, const char *path);

// Read an index from disk and attach it to an existing vector store.
//
// vs : must contain the same vectors, in the same order, that the index
//      was built over. The index stores node IDs, which are indices into
//      this store — a mismatch produces silently wrong results.
//
// Returns NULL on failure.
HNSW *hnsw_load(const char *path, VectorStore *vs);

// hnsw.h
// Total bytes allocated by this index, excluding the borrowed vectors.
size_t hnsw_memory_bytes(const HNSW *h);

int hnsw_max_level(const HNSW *h);

// How many nodes actually belong to this layer.
int hnsw_layer_members(const HNSW *h, int layer);

// hnsw.h
int hnsw_get_M(const HNSW *h);

// hnsw.h
// Degree of a node in a given layer. Returns 0 for invalid layers.
int hnsw_degree(HNSW *h, int node, int layer);
#endif