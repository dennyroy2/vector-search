#ifndef GRAPH_H
#define GRAPH_H

#include "vectors.h"
#include "visited.h"
#include "heap.h"

typedef struct {
    int * neighbours; // n * M ints
    int * degrees; // array : ith index represents number of neighbours for ith node
    int n; // number of nodes
    int M; // max neighbours per node
} Graph;

// Allocate a graph with no edges. Returns NULL on allocation failure
Graph *graph_create(int n, int M);

void graph_free(Graph *g);

// Add a DIRECTED edge from -> to. Returns 1 on success, 0 if `from` is full.
// Callers wanting an undirected edge must call this twice.
int graph_add_edge(Graph *g, int from, int to);

// Pointer to a node's neighbour list. Writes the valid count to out_count.
// The pointer stays valid until the graph is modified or freed.
const int *graph_neighbours(const Graph *g, int node, int *out_count);

// How many neighbours a node currently has.
int graph_degree(const Graph *g, int node);

int graph_fill_random( Graph * g, int seed);

// Greedy hill-climbing search. Walks from `entry` toward `query`,
// always moving to the closest neighbour, stopping when none improves.
//
// Returns the node id of the local minimum reached.
// Writes its squared distance to *out_dist.
// Writes the number of distance computations performed to *out_hops —
// this is the number that justifies the whole approach, so measure it.
int graph_greedy_search(const Graph *g, const VectorStore *vs, const float *query, int entry, VisitedSet *visited, 
                        float *out_dist, int *out_ndists, int * out_hops);

// Copy node's neighbour list into `out` (caller-allocated, >= M ints).
// Returns the number written. Lets Python inspect the graph.
int graph_get_neighbours_copy(const Graph *g, int node, int *out);

// Beam search: explores the graph keeping the `ef` best candidates found.
//
// ef        : beam width. Must be >= k. Larger explores more: higher
//             recall, lower throughput. This is THE tuning knob.
// out_ids   : caller-allocated, >= k ints
// out_dists : caller-allocated, >= k floats
//
// Returns the number written, ascending by distance.
int graph_beam_search(const Graph *g, const VectorStore *vs,
                      const float *query, int entry, int ef, int k,
                      VisitedSet *visited,
                      int *out_ids, float *out_dists, int *out_ndists,
                    MaxHeap * candidates, MaxHeap * results);

// Overwrite a specific neighbour slot. No degree change.
int graph_replace_edge(Graph *g, int node, int slot, int new_neighbour);

// graph.h

// Build a graph by inserting every vector one at a time, connecting each
// to its M nearest already-inserted neighbours.
//
// ef_construction : beam width during insertion. Higher = better graph,
//                   slower build. Paid once, unlike efSearch.
//
// Returns 1 on success, 0 on allocation failure.
int graph_build(Graph *g, const VectorStore *vs, int ef_construction);

#endif