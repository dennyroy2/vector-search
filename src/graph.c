#include <stdlib.h>
#include "graph.h"
#include "vectors.h"
#include "visited.h"
#include "distance.h"
#include "heap.h"

Graph *graph_create(int n, int M) {
    Graph * graph = malloc(sizeof(Graph));
    if (!graph) return NULL;

    graph->neighbours = malloc((size_t)sizeof(int) * n * M);

    if (!graph->neighbours) {free(graph); return NULL;}

    graph->degrees = calloc(n, sizeof(int));

    if (!graph->degrees) {
        free(graph->neighbours);
        free(graph);
        return NULL;
    }
    graph->M = M;
    graph->n = n;
    return graph;
}

void graph_free(Graph *g) {
    if (!g) return;
    free(g->degrees);
    free(g->neighbours);
    free(g);
}

int graph_add_edge(Graph *g, int from, int to) {
    int count = g->degrees[from];
    if (count >= g->M) {return 0;}

    *(g->neighbours + (g->M * from) + count) = to;
    g->degrees[from] = count + 1;
    return 1;
}

const int *graph_neighbours(const Graph *g, int node, int *out_count) {
    *out_count = g->degrees[node];
    const int * ptr = &g->neighbours[node * g->M];
    return ptr;
}

int graph_degree(const Graph *g, int node) {
    return g->degrees[node];
}

int graph_fill_random( Graph * g, int seed) {
    if (g->M >= g->n) {return 0;}
    int n = g->n;
    int M = g->M;
    srand(seed);

    for (int i = 0; i < n; i++) {
        int alloted = graph_degree(g, i);

        while (alloted < M) {
            int candidate = rand() % n;
            if (candidate == i) continue;

            int already = 0;
            for (int j = 0; j < graph_degree(g, i); j++) {
                if (g->neighbours[i*M + j] == candidate) {
                    already = 1;
                    break;
                }
            }
            if (already) continue;

            graph_add_edge(g, i, candidate);
            graph_add_edge(g, candidate, i);
            alloted++;
        }
    }
    return 1;
}

int graph_greedy_search(const Graph *g, const VectorStore *vs, const float *query, int entry, VisitedSet *visited, 
                        float *out_dist, int *out_ndists, int * out_hops) {
        visited_reset(visited);
        int hops = 0;
        int current = entry;
        visited_mark(visited, current);
        float current_dist = l2sq_distance(vs_get(vs, current), query, vs->dim);
        int distance_counter = 1;

        while (1) {
            int best = -1;
            float best_dist = current_dist;
            int out_count;
            const int * ptr = graph_neighbours(g, current, &out_count);

            for (int i = 0; i < out_count; i++) {
                if (visited_check(visited, ptr[i]) == 1) continue;

                visited_mark(visited, ptr[i]);
                float new_dist = l2sq_distance(vs_get(vs, ptr[i]), query, vs->dim);
                distance_counter++;

                if (new_dist < best_dist) {
                    hops++;
                    best_dist = new_dist;
                    best = ptr[i];
                }
            }
            if (best == -1) break;

            current = best;
            current_dist = best_dist;

        }
  
    *out_ndists = distance_counter;
    *out_dist = current_dist;
    *out_hops = hops;
    return current;
    }

int graph_get_neighbours_copy(const Graph *g, int node, int *out) {
    int count;
    const int *nbrs = graph_neighbours(g, node, &count);
    for (int j = 0; j < count; j++) out[j] = nbrs[j];
    return count;
}

int graph_beam_search(const Graph *g, const VectorStore *vs,
                      const float *query, int entry, int ef, int k,
                      VisitedSet *visited,
                      int *out_ids, float *out_dists, int *out_ndists) {

    if ((ef <= 0) || (k <= 0)) return 0;

    if (ef < k) { ef = k;}

    MaxHeap * candidates = heap_create(g->n, 0);
    MaxHeap * results = heap_create(ef+1, 1);
    if ((candidates == NULL) || (results == NULL)) return 0;

    visited_reset(visited);
    visited_mark(visited, entry);
    int dist_counter = 1;

    float dist = l2sq_distance(vs_get(vs, entry), query, vs->dim);
    heap_push(candidates, entry, dist);
    heap_push(results, entry, dist);


    while (heap_size(candidates) > 0) {
        Candidate c, worst;
        heap_pop(candidates, &c);
        heap_peek(results, &worst);

        if (heap_size(results) >= ef && c.dist > worst.dist) break;

        int out_count;
        const int * ptr = graph_neighbours(g, c.id, &out_count);

        for (int i = 0; i < out_count; i++) {
            if (visited_check(visited, ptr[i]) == 1) continue;
            dist_counter++;

            visited_mark(visited, ptr[i]);
            float current_dist = l2sq_distance(vs_get(vs, ptr[i]), query, vs->dim);
            heap_peek(results, &worst);

            if (heap_size(results) < ef || worst.dist > current_dist) {
                heap_push(candidates, ptr[i], current_dist);
                heap_push(results, ptr[i], current_dist);

                Candidate throw;
                if (heap_size(results) > ef) {heap_pop(results, &throw);}
            }
        }
    }

    Candidate out;
    int count = heap_size(results);
    int n_out = count < k ? count : k;
    for (int i = count - 1 ; i >= 0; i--) {
        heap_pop(results, &out);
        if (i < n_out) {
            out_ids[i] = out.id;
            out_dists[i] = out.dist;
        }
    }

    *out_ndists = dist_counter;
    heap_free(candidates);
    heap_free(results);
    return n_out;
}