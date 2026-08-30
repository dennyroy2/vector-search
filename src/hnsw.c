#include "hnsw.h"
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <stddef.h>

int hnsw_random_level(double mL) {
    double u = (rand() + 1.0)/ (RAND_MAX + 1.0);

    int level = (int)(-log(u) * mL);

    if (level > HNSW_MAX_LEVEL) level = HNSW_MAX_LEVEL;
    return level;
}

static inline double hnsw_default_mL(int M) {
    assert(M > 1);
    return 1/log(M);
}

HNSW * hnsw_create(VectorStore * vs, int M) {
    HNSW * h = malloc(sizeof(HNSW));
    if (!h) return NULL;
    h->vs = vs;
    h->layers = calloc(HNSW_MAX_LEVEL + 1, sizeof(Graph *));
    h->node_levels = calloc(vs->n, sizeof(int));
    h->max_level = 0;
    h->entry_point = -1;
    h->n = vs->n;
    h->M = M;
    h->M0 = 2*M;
    h->mL = hnsw_default_mL(M);
    if ((!h->layers) || (!h->node_levels)) {
        free(h->layers);
        free(h->node_levels);
        return NULL;
    }
    h->layers[0] = graph_create(h->n, h->M0);
    return h;
}

int hnsw_ensure_level(HNSW * h, int target) {
    if (target > HNSW_MAX_LEVEL) return 0;
    for (int i = 1; i <= target; i++) {
        if (i > h->max_level) {
            if (!(h->layers[i] = graph_create(h->n, h->M))) return 0;
            h->max_level++;
        }
    }
    return 1;
}

void hnsw_free(HNSW * h){
    if (h) {
    for (int i = 0; i <= h->max_level; i++) {
        graph_free(h->layers[i]);
    }
    free(h->layers);
    free(h->node_levels);
    free(h);
}
}

int hnsw_node_in_layer(const HNSW *h, int node, int layer) {
    return (h->node_levels[node] >= layer);
}

Graph * hnsw_layer(HNSW * h, int layer) {
    if (layer <= h->max_level && layer >= 0) return h->layers[layer];
    return NULL;
}



int hnsw_insert(HNSW *h, int node, int ef_construction,
                VisitedSet *visited, MaxHeap *candidates, MaxHeap *results,
                int *found_ids, float *found_dists, int *selected) {

    int prev_max = h->max_level;
    int level = hnsw_random_level(h->mL);
    h->node_levels[node] = level;

    if (!(hnsw_ensure_level(h, level))) return 0;
   
    if (h->entry_point == -1) {
        h->entry_point = node;
        return 1;
    }

    int cur = h->entry_point;
    int out_ndists;
    for (int l = prev_max; l > level; l--) {
        if (graph_beam_search(hnsw_layer(h, l), h->vs, vs_get(h->vs, node), cur, 1, 1, visited, found_ids, 
                                found_dists, &out_ndists, candidates, results)) cur = found_ids[0];
    }

    int l;
    l = level < prev_max ? level : prev_max;
    int n_sel;
    for (; l >= 0; l--) {
        int n_cand = graph_beam_search(hnsw_layer(h, l), h->vs, vs_get(h->vs, node), cur, ef_construction, ef_construction,
                                visited, found_ids, found_dists, &out_ndists, candidates, results);
        if (n_cand) cur = found_ids[0];
        if (l == 0) {
            n_sel = graph_select_neighbours(h->vs, node, found_ids, found_dists, n_cand, h->M0, selected);
            if (node == 5000) {
                    printf("  node 5000 layer %d: n_cand=%d, n_sel=%d\n", l, n_cand, n_sel);
                    fflush(stdout);
                }
        } else {
            n_sel = graph_select_neighbours(h->vs, node, found_ids, found_dists, n_cand, h->M, selected);
        }
        for (int c = 0; c < n_sel; c++) {
            graph_add_edge(hnsw_layer(h, l), node, selected[c]);

            if (!graph_add_edge(hnsw_layer(h, l), selected[c], node)) {
                int count;
                const int *neighbours = graph_neighbours(hnsw_layer(h, l), selected[c], &count);
                const float * vec = vs_get(h->vs, selected[c]);

                int worst_slot = -1;
                float worst_dist = -1.0f;
                for (int k = 0; k < count; k++) {
                    float current_dist = l2sq_distance(vs_get(h->vs, neighbours[k]), vec, h->vs->dim);
                    if (current_dist > worst_dist) {
                        worst_dist = current_dist;
                        worst_slot = k;
                    }
                }
             if (worst_dist > l2sq_distance(vec, vs_get(h->vs, node), h->vs->dim)) {
                    graph_replace_edge(hnsw_layer(h, l), selected[c], worst_slot, node);
                }
        }
    }

}
    if (level > prev_max) h->entry_point = node;
    return 1;
                }

int hnsw_build(HNSW *h, int ef_construction, int seed) {
    
    srand(seed);

    VisitedSet * visited = visited_create(h->n);
    MaxHeap * candidates = heap_create(h->n, 0);
    MaxHeap * results = heap_create(ef_construction+1, 1);
    int * found_ids = calloc(ef_construction, sizeof(int));
    float * found_dists = calloc(ef_construction, sizeof(float));
    int * selected = calloc(h->M0, sizeof(int));

    if (!found_ids || !found_dists) {free(found_ids); free(found_dists); free(selected); return 0;}
    if (!selected) {free(found_ids); free(found_dists); free(selected); return 0;}

    for (int i = 0; i < h->n; i++) {
        if(!hnsw_insert(h, i, ef_construction, visited, candidates, results, found_ids, found_dists, selected)) return 0;
    }
    free(found_ids); 
    free(found_dists); 
    free(selected);
    heap_free(candidates);
    heap_free(results);
    visited_free(visited);
    return 1;
}

// hnsw.c

int hnsw_search(HNSW *h, const float *query, int k, int ef_search,
                VisitedSet *visited, MaxHeap *candidates, MaxHeap *results,
                int *out_ids, float *out_dists, int *out_ndists, int * out_descent_ndists) {

    if (h->entry_point == -1) return 0;
   
    int cur = h->entry_point;
    int ndists = 0;
    int descent = 0;
    int num_results = 0;

    for (int l = h->max_level; l > 0; l--) {
        num_results = graph_beam_search(hnsw_layer(h, l), h->vs, query, cur, 1, 1, visited, out_ids, out_dists, out_ndists,
                        candidates, results);
        if (num_results) {
        //printf("l is %d, search returns %d results, and ndists is %d\n", l, num_results, *out_ndists);
        cur = out_ids[0];
        descent += *out_ndists;
        }
    }
    ndists += descent;
    num_results = graph_beam_search(hnsw_layer(h, 0), h->vs, query, cur, ef_search, k, visited, out_ids, out_dists, out_ndists, candidates, results);
    if (num_results) {
        ndists += *out_ndists;
    }

    *out_ndists = ndists;
    *out_descent_ndists = descent;
    return num_results;
}

int hnsw_save (HNSW * h, const char * path) {

    FILE * f = fopen(path, "wb");

    if (!f) return 0;

    int magic = HNSW_MAGIC;
    int version = HNSW_VERSION;

    if(!fwrite(&magic, sizeof(int), 1, f)) { fclose(f); return 0;}
    if(!fwrite(&version, sizeof(int), 1, f)) { fclose(f); return 0;}
    if(!fwrite(&h->n, sizeof(int), 1, f)) { fclose(f); return 0;}
    if(!fwrite(&h->M, sizeof(int), 1, f)) { fclose(f); return 0;}
    if(!fwrite(&h->M0, sizeof(int), 1, f)) { fclose(f); return 0;}
    if(!fwrite(&h->max_level, sizeof(int), 1, f)) { fclose(f); return 0;}
    if(!fwrite(&h->entry_point, sizeof(int), 1, f)) { fclose(f); return 0;}
    if(!fwrite(&h->vs->dim, sizeof(int), 1, f)) { fclose(f); return 0;}
    if(!fwrite(&h->mL, sizeof(double), 1, f)) { fclose(f); return 0;}

    if(fwrite(h->node_levels, sizeof(int), h->n, f) != (size_t)h->n) { fclose(f); return 0;}

    for (int l = 0; l <= h->max_level; l++){
        if(fwrite(hnsw_layer(h, l)->degrees, sizeof(int), h->n, f) != (size_t)h->n) { fclose(f); return 0;}
        if(fwrite(hnsw_layer(h, l)->neighbours, sizeof(int), h->n * hnsw_layer(h, l)->M, f) != (size_t)h->n*hnsw_layer(h, l)->M) { fclose(f); return 0;}
    }

    fclose(f);
    return 1;
}

HNSW *hnsw_load(const char *path, VectorStore *vs) {

    FILE * f = fopen(path, "rb");
    if (!f) return NULL;
    int magic, version;

    if (fread(&magic, sizeof(int), 1, f) != 1) {fclose(f); return NULL;}
    if (magic != HNSW_MAGIC) {fclose(f); return NULL;}

    if (fread(&version, sizeof(int), 1, f) != 1) {fclose(f); return NULL;}
    if (version != HNSW_VERSION) {fclose(f); return NULL;}

    int n, M, M0, max_level, entry_point, dim;
    double mL;

    if (fread(&n, sizeof(int), 1, f) != 1) {fclose(f); return NULL;}
    if (fread(&M, sizeof(int), 1, f) != 1) {fclose(f); return NULL;}
    if (fread(&M0, sizeof(int), 1, f) != 1) {fclose(f); return NULL;}
    if (fread(&max_level, sizeof(int), 1, f) != 1) {fclose(f); return NULL;}
    if (fread(&entry_point, sizeof(int), 1, f) != 1) {fclose(f); return NULL;}
    if (fread(&dim, sizeof(int), 1, f) != 1) {fclose(f); return NULL;}
    if (fread(&mL, sizeof(double), 1, f) != 1) {fclose(f); return NULL;}

    if ((n != vs-> n) || (dim != vs->dim)) {fclose(f); return NULL;}
    if ((max_level < 0 || max_level > HNSW_MAX_LEVEL)) {fclose(f); return NULL;}
    if (M < 0 || M0 < 0) {fclose(f); return NULL;}

    
    HNSW * h = hnsw_create(vs, M);
    if (!h) {fclose(f); return NULL;}

    if (fread(h->node_levels, sizeof(int), n, f) != (size_t)n) {fclose(f); hnsw_free(h); return NULL;}

    hnsw_ensure_level(h, max_level);

    h->mL = mL;
    h->entry_point = entry_point;

    for (int l = 0; l <= max_level; l++) {
        if (fread(hnsw_layer(h, l)->degrees, sizeof(int), hnsw_layer(h, l)->n, f) != (size_t)h->n) {fclose(f); hnsw_free(h); return NULL;}
        if (fread(hnsw_layer(h, l)->neighbours, sizeof(int),  h->n * hnsw_layer(h, l)->M, f) != (size_t)hnsw_layer(h, l)->n * hnsw_layer(h, l)->M) {fclose(f); hnsw_free(h);return NULL;}
    }
    fclose(f);
    return h;
}

size_t hnsw_memory_bytes(const HNSW *h) {
    size_t size = 0;
    size += sizeof(HNSW); //struct itself
    size += (HNSW_MAX_LEVEL+1) * sizeof(Graph *); //layers array
    size += h->vs->n * sizeof(int); //node_levels

    for (int l = 1; l <= h->max_level; l++) {
        size += sizeof(Graph) + (h->n * sizeof(int)) + (h->n * h->M * sizeof(int));
    }
    size += sizeof(Graph) + (h->n * sizeof(int)) + (h->M0 * h->n * sizeof(int));

    return size;
}

int hnsw_max_level(const HNSW *h) {
    return h->max_level;
}

int hnsw_layer_members(const HNSW *h, int layer) {
    int members = 0;
    for (int i = 0; i < h->n; i++) {
        if (h->node_levels[i] >= layer) members++;
    }
    return members;
}

int hnsw_get_M(const HNSW *h) { return h->M; }

// hnsw.c
int hnsw_degree(HNSW *h, int node, int layer) {
    Graph *g = hnsw_layer(h, layer);
    if (!g || node < 0 || node >= h->n) return 0;
    return graph_degree(g, node);
}