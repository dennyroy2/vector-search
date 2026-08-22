#include <stdlib.h>
#include "graph.h"
#include "vectors.h"
#include "visited.h"
#include "distance.h"

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