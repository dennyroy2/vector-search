#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "hnsw.h"
#include "vectors.h"

// --- Case 1: first insertion into an empty index ----------------------
static void test_first_insert(void) {
    int n = 10, M = 4;
    float data[] = {0,1,2,3,4,5,6,7,8,9};
    VectorStore *vs = vs_create(data, n, 1);
    HNSW *h = hnsw_create(vs, M);

    assert(h->entry_point == -1);

    srand(1);
    VisitedSet *v = visited_create(n);
    MaxHeap *cand = heap_create(n, 0);
    MaxHeap *res  = heap_create(33, 1);
    int   fids[32];
    float fdst[32];
    int   sel[8];

    assert(hnsw_insert(h, 0, 32, v, cand, res, fids, fdst, sel) == 1);

    // TODO: assert entry_point is now 0
    assert(h->entry_point == 0);
    // TODO: assert max_level equals node_levels[0]
    assert(h->max_level == h->node_levels[0]);
    // TODO: assert node 0 has degree 0 in layer 0 — nothing to connect to
    assert(graph_degree(hnsw_layer(h, 0), 0) == 0);

    visited_free(v); heap_free(cand); heap_free(res);
    hnsw_free(h); vs_free(vs);
    printf("  first_insert: PASS\n");
}

// --- Case 2: two nodes must connect in layer 0 ------------------------
static void test_two_nodes(void) {
    int n = 10, M = 4;
    float data[] = {0,1,2,3,4,5,6,7,8,9};
    VectorStore *vs = vs_create(data, n, 1);
    HNSW *h = hnsw_create(vs, M);

    srand(1);
    VisitedSet *v = visited_create(n);
    MaxHeap *cand = heap_create(n, 0);
    MaxHeap *res  = heap_create(33, 1);
    int   fids[32]; float fdst[32]; int sel[8];

    hnsw_insert(h, 0, 32, v, cand, res, fids, fdst, sel);
    hnsw_insert(h, 1, 32, v, cand, res, fids, fdst, sel);

    // TODO: assert both nodes have degree 1 in layer 0
    assert(graph_degree(hnsw_layer(h, 0), 0) == 1);
    assert(graph_degree(hnsw_layer(h, 0), 1) == 1);
    // TODO: assert node 0's only neighbour is 1, and vice versa
    //       Use graph_neighbours on hnsw_layer(h, 0).
    int out_count;
    const int * nbrs0 = graph_neighbours(hnsw_layer(h, 0), 0, &out_count);
    const int * nbrs1 = graph_neighbours(hnsw_layer(h, 0), 1, &out_count);
    assert(*nbrs0 == 1);
    assert(*nbrs1 == 0);

    visited_free(v); heap_free(cand); heap_free(res);
    hnsw_free(h); vs_free(vs);
    printf("  two_nodes: PASS\n");
}

// --- Case 3: full build, structural invariants ------------------------
static void test_build_structure(void) {
    int n = 2000, M = 8;
    float *data = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) data[i] = (float)(i * 3 % n);

    VectorStore *vs = vs_create(data, n, 1);
    HNSW *h = hnsw_create(vs, M);
    assert(hnsw_build(h, 64, 42) == 1);

    // Every node must have somewhere to go in layer 0.
    int isolated = 0;
    for (int i = 0; i < n; i++) {
        if (graph_degree(hnsw_layer(h, 0), i) == 0) isolated++;
    }
    printf("    isolated in layer 0: %d of %d\n", isolated, n);
    assert(isolated <= 1);   // node 0 entered an empty index; later nodes
                             // should have connected back to it

    // The entry point must sit at the top of the hierarchy. This is the
    // assertion that catches the ensure_level ordering bug — if the entry
    // point never updates, it stays at node 0, which is almost certainly
    // level 0 while max_level is 2 or 3.
    printf("    entry_point %d at level %d, max_level %d\n",
           h->entry_point, h->node_levels[h->entry_point], h->max_level);
    assert(h->node_levels[h->entry_point] == h->max_level);

    for (int l = 0; l <= h->max_level; l++) {
        Graph *g = hnsw_layer(h, l);
        assert(g != NULL);

        for (int i = 0; i < n; i++) {
            int count;
            const int *nbrs = graph_neighbours(g, i, &count);

            // A node with edges in layer l must belong to layer l.
            // Catches edges written into the wrong layer.
            assert(count == 0 || hnsw_node_in_layer(h, i, l));
            assert(count <= g->M);

            for (int j = 0; j < count; j++) {
                assert(nbrs[j] != i);                  // no self-loop
                assert(nbrs[j] >= 0 && nbrs[j] < n);   // valid id

                // The neighbour must also live in this layer.
                assert(hnsw_node_in_layer(h, nbrs[j], l));

                for (int k = j + 1; k < count; k++)
                    assert(nbrs[k] != nbrs[j]);        // no duplicate
            }
        }
    }

    free(data); hnsw_free(h); vs_free(vs);
    printf("  build_structure: PASS\n");
}

// --- Case 4: layer occupancy matches the level distribution -----------
static void test_layer_occupancy(void) {
    int n = 20000, M = 16;
    float *data = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) data[i] = (float)(i * 7 % n);

    VectorStore *vs = vs_create(data, n, 1);
    HNSW *h = hnsw_create(vs, M);
    assert(hnsw_build(h, 32, 7) == 1);

    printf("    max_level = %d, entry_point = %d (level %d)\n",
           h->max_level, h->entry_point, h->node_levels[h->entry_point]);

    int prev = 0;
    for (int l = 0; l <= h->max_level; l++) {
        int members = 0;
        for (int i = 0; i < n; i++) {
            if (hnsw_node_in_layer(h, i, l)) members++;
        }

        printf("    layer %d: %6d members", l, members);
        if (l > 0 && members > 0) {
            printf("   (1/%.1f of layer %d)", (double)prev / members, l - 1);
        }
        printf("\n");

        if (l == 0) {
            assert(members == n);       // layer 0 holds everything
        } else {
            assert(members < prev);     // strictly shrinking upward
        }

        // Layer 1 should hold roughly n/M. Wide band — this is a random
        // draw, and the point is confirming the order of magnitude.
        if (l == 1) {
            assert(members > n / (4 * M));
            assert(members < n * 4 / M);
        }
        prev = members;
    }

    free(data); hnsw_free(h); vs_free(vs);
    printf("  layer_occupancy: PASS\n");
}

// --- Case 5: layer 0 connectivity -------------------------------------
static void test_layer0_connectivity(void) {
    int n = 5000, M = 8;
    float *data = malloc(n * 2 * sizeof(float));
    srand(11);
    // Two clusters — the case that broke naive construction at step 21.
    for (int i = 0; i < n; i++) {
        int c = (i < n / 2) ? 0 : 1;
        data[i*2]     = c * 1000.0f + (float)(rand() % 50);
        data[i*2 + 1] = c * 1000.0f + (float)(rand() % 50);
    }

    VectorStore *vs = vs_create(data, n, 2);
    HNSW *h = hnsw_create(vs, M);
    assert(hnsw_build(h, 64, 3) == 1);

    int *seen  = calloc(n, sizeof(int));
    int *queue = malloc(n * sizeof(int));
    int head = 0, tail = 0;
    seen[h->entry_point] = 1;
    queue[tail++] = h->entry_point;

    while (head < tail) {
        int node = queue[head++];
        int count;
        const int *nbrs = graph_neighbours(hnsw_layer(h, 0), node, &count);
        for (int j = 0; j < count; j++) {
            if (!seen[nbrs[j]]) { seen[nbrs[j]] = 1; queue[tail++] = nbrs[j]; }
        }
    }

    int reached = 0;
    for (int i = 0; i < n; i++) reached += seen[i];
    printf("    layer 0 reachable: %d of %d (%.1f%%)\n",
           reached, n, 100.0 * reached / n);
    assert(reached >= n * 95 / 100);

    free(seen); free(queue); free(data);
    hnsw_free(h); vs_free(vs);
    printf("  layer0_connectivity: PASS\n");
}

// --- Case 6: determinism ----------------------------------------------
static void test_deterministic(void) {
    int n = 1000, M = 8;
    float *data = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) data[i] = (float)(i * 13 % n);
    VectorStore *vs = vs_create(data, n, 1);

    HNSW *a = hnsw_create(vs, M);
    HNSW *b = hnsw_create(vs, M);
    assert(hnsw_build(a, 32, 99) == 1);
    assert(hnsw_build(b, 32, 99) == 1);

    assert(a->max_level == b->max_level);
    assert(a->entry_point == b->entry_point);

    for (int i = 0; i < n; i++) {
        assert(a->node_levels[i] == b->node_levels[i]);
    }

    for (int l = 0; l <= a->max_level; l++) {
        Graph *ga = hnsw_layer(a, l);
        Graph *gb = hnsw_layer(b, l);
        for (int i = 0; i < n; i++) {
            int ca, cb;
            const int *na = graph_neighbours(ga, i, &ca);
            const int *nb = graph_neighbours(gb, i, &cb);
            assert(ca == cb);
            for (int j = 0; j < ca; j++) assert(na[j] == nb[j]);
        }
    }

    free(data); hnsw_free(a); hnsw_free(b); vs_free(vs);
    printf("  deterministic: PASS\n");
}

int main(void) {
    printf("test_hnsw_insert:\n");
    test_first_insert();
    test_two_nodes();
    test_build_structure();
    test_layer_occupancy();
    test_layer0_connectivity();
    test_deterministic();
    printf("test_hnsw_insert: all passed\n");
    return 0;
}