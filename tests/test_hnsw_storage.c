#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "hnsw.h"
#include "vectors.h"

// --- Case 1: a fresh index has layer 0 and nothing else ---------------
static void test_create(void) {
    int n = 1000, M = 16;
    float *data = calloc(n, sizeof(float));
    VectorStore *vs = vs_create(data, n, 1);

    HNSW *h = hnsw_create(vs, M);
    assert(h != NULL);

    // TODO: assert max_level is 0
    assert(h->max_level == 0);
    // TODO: assert entry_point is -1 (nothing inserted)
    assert(h->entry_point == -1);
    // TODO: assert M and M0 are set, with M0 == 2*M
    assert(h->M0 == h->M*2);
    // TODO: assert n was copied from the vector store
    assert(h->n == vs->n);
    // TODO: assert layer 0 exists and its graph has M == M0 (not M)
    //       Reach in: hnsw_layer(h, 0)->M
    assert(hnsw_layer(h, 0)->M == 2*M);
    // TODO: assert layers 1 through HNSW_MAX_LEVEL are all NULL
    for (int i = 1; i <= HNSW_MAX_LEVEL; i++) {
        assert(hnsw_layer(h, i) == NULL);
    }
    // TODO: assert every node_levels entry starts at 0
    for (int i = 0; i < n; i++) {
        assert(h->node_levels[i] == 0);
    }
    hnsw_free(h); vs_free(vs); free(data);
    printf("  create: PASS\n");
}

// --- Case 2: growing the hierarchy ------------------------------------
static void test_ensure_level(void) {
    int n = 500, M = 8;
    float *data = calloc(n, sizeof(float));
    VectorStore *vs = vs_create(data, n, 1);
    HNSW *h = hnsw_create(vs, M);

    assert(hnsw_ensure_level(h, 3) == 1);

    // TODO: assert max_level is now 3
    assert(h->max_level == 3);
    // TODO: assert layers 1, 2, 3 all exist
    assert(hnsw_layer(h, 1) != NULL);
    assert(hnsw_layer(h, 2) != NULL);
    assert(hnsw_layer(h, 3) != NULL);
    // TODO: assert each of them has M == h->M (8), NOT M0 (16).
    assert(hnsw_layer(h, 1)->M == h->M);
    assert(hnsw_layer(h, 2)->M == h->M);
    assert(hnsw_layer(h, 3)->M == h->M);
    //       Only layer 0 gets the doubled connection count.
    // TODO: assert layer 4 is still NULL
    assert(hnsw_layer(h, 4) == NULL);

    hnsw_free(h); vs_free(vs); free(data);
    printf("  ensure_level: PASS\n");
}

// --- Case 3: ensure_level is idempotent and never shrinks -------------
// Insertion calls this for every node's level, so it's called constantly
// with targets below max_level. Those calls must be no-ops.
static void test_ensure_level_idempotent(void) {
    int n = 100, M = 8;
    float *data = calloc(n, sizeof(float));
    VectorStore *vs = vs_create(data, n, 1);
    HNSW *h = hnsw_create(vs, M);

    hnsw_ensure_level(h, 3);
    Graph *layer2_before = hnsw_layer(h, 2);

    // TODO: call ensure_level with 1, then 0, then 3 again.
    hnsw_ensure_level(h, 1);
    assert(h->max_level == 3);
    hnsw_ensure_level(h, 0);
    assert(h->max_level == 3);
    hnsw_ensure_level(h, 3);
    assert(h->max_level == 3);
    // TODO: assert max_level is still 3 each time.
    // TODO: assert hnsw_layer(h, 2) returns the SAME pointer as before —
    //       the layer must not be reallocated, which would leak the old
    //       one and discard its edges.
    assert(hnsw_layer(h, 2) == layer2_before);

    hnsw_free(h); vs_free(vs); free(data);
    printf("  ensure_level_idempotent: PASS\n");
}

// --- Case 4: out-of-range targets are rejected ------------------------
static void test_ensure_level_bounds(void) {
    int n = 100, M = 8;
    float *data = calloc(n, sizeof(float));
    VectorStore *vs = vs_create(data, n, 1);
    HNSW *h = hnsw_create(vs, M);

    // TODO: assert ensure_level(h, HNSW_MAX_LEVEL) succeeds
    assert(hnsw_ensure_level(h, HNSW_MAX_LEVEL) == 1);
    // TODO: assert ensure_level(h, HNSW_MAX_LEVEL + 1) returns 0
    assert(hnsw_ensure_level(h, HNSW_MAX_LEVEL+1) == 0);
    // TODO: assert max_level did not exceed HNSW_MAX_LEVEL
    assert(h->max_level <= HNSW_MAX_LEVEL);
    hnsw_free(h); vs_free(vs); free(data);
    printf("  ensure_level_bounds: PASS\n");
}

// --- Case 5: membership is about node_levels, not degree --------------
// A node can belong to a layer and have no neighbours there yet — that's
// its state immediately after insertion. Testing membership via degree
// would report it absent.
static void test_membership(void) {
    int n = 100, M = 8;
    float *data = calloc(n, sizeof(float));
    VectorStore *vs = vs_create(data, n, 1);
    HNSW *h = hnsw_create(vs, M);
    hnsw_ensure_level(h, 3);

    // Set levels by hand — insertion isn't written yet.
    h->node_levels[5]  = 0;
    h->node_levels[10] = 2;
    h->node_levels[20] = 3;

    // Node 5 is level 0: layer 0 only.
    assert(hnsw_node_in_layer(h, 5, 0));
    assert(!hnsw_node_in_layer(h, 5, 1));
    assert(!hnsw_node_in_layer(h, 5, 2));

    // Node 10 is level 2: layers 0, 1 AND 2. A node at level L appears in
    // every layer from 0 up to L, not only in layer L.
    assert(hnsw_node_in_layer(h, 10, 0));
    assert(hnsw_node_in_layer(h, 10, 1));
    assert(hnsw_node_in_layer(h, 10, 2));
    assert(!hnsw_node_in_layer(h, 10, 3));

    // Node 20 is level 3: all four layers.
    assert(hnsw_node_in_layer(h, 20, 0));
    assert(hnsw_node_in_layer(h, 20, 1));
    assert(hnsw_node_in_layer(h, 20, 2));
    assert(hnsw_node_in_layer(h, 20, 3));

    // THE assertion: node 20 has no edges anywhere, yet belongs to every
    // layer up to 3. A degree-based membership test would report it absent
    // from all of them.
    assert(graph_degree(hnsw_layer(h, 3), 20) == 0);
    assert(hnsw_node_in_layer(h, 20, 3));

    hnsw_free(h); vs_free(vs); free(data);
    printf("  membership: PASS\n");
}

// --- Case 6: hnsw_layer bounds ----------------------------------------
static void test_layer_accessor(void) {
    int n = 100, M = 8;
    float *data = calloc(n, sizeof(float));
    VectorStore *vs = vs_create(data, n, 1);
    HNSW *h = hnsw_create(vs, M);
    hnsw_ensure_level(h, 2);

    assert(hnsw_layer(h, 0) != NULL);
    assert(hnsw_layer(h, 1) != NULL);
    assert(hnsw_layer(h, 2) != NULL);

    // Allocated slot in the pointer array, but no layer created there.
    assert(hnsw_layer(h, 3) == NULL);

    // Out of range entirely — must be rejected before indexing, or this
    // reads past the end of the layers array.
    assert(hnsw_layer(h, 999) == NULL);
    assert(hnsw_layer(h, -1) == NULL);

    hnsw_free(h); vs_free(vs); free(data);
    printf("  layer_accessor: PASS\n");
}

// --- Case 7: free on a partially-grown index, and on NULL -------------
static void test_free(void) {
    int n = 100, M = 8;
    float *data = calloc(n, sizeof(float));
    VectorStore *vs = vs_create(data, n, 1);

    // Grown to different heights — free must handle all of them without
    // leaking allocated layers or touching unallocated ones.
    for (int target = 0; target <= 5; target++) {
        HNSW *h = hnsw_create(vs, M);
        assert(h != NULL);
        if (target > 0) assert(hnsw_ensure_level(h, target) == 1);
        hnsw_free(h);
    }

    hnsw_free(NULL);   // must not crash

    vs_free(vs); free(data);
    printf("  free: PASS\n");
}

// --- Case 8: memory footprint report ----------------------------------
// Not an assertion — the number that justifies full-width layers. Print it
// so the tradeoff is documented rather than assumed.
static void test_memory_report(void) {
    int n = 10000, M = 16;
    float *data = calloc(n * 128, sizeof(float));
    VectorStore *vs = vs_create(data, n, 128);
    HNSW *h = hnsw_create(vs, M);
    hnsw_ensure_level(h, 3);

    // Each layer: n*M ints for neighbours, plus n ints for degrees.
    size_t layer0 = (size_t)n * h->M0 * sizeof(int) + (size_t)n * sizeof(int);
    size_t upper  = (size_t)n * h->M  * sizeof(int) + (size_t)n * sizeof(int);
    size_t total  = layer0 + 3 * upper;

    printf("    n=%d M=%d, 4 layers\n", n, M);
    printf("    layer 0:  %.2f MB\n", layer0 / 1e6);
    printf("    layers 1-3: %.2f MB each\n", upper / 1e6);
    printf("    graph total: %.2f MB\n", total / 1e6);
    printf("    vectors:     %.2f MB\n", (size_t)n * 128 * sizeof(float) / 1e6);

    // TODO: extrapolate to n=1,000,000 and print that too. That's the
    //       number your decisions-log entry claims — verify it.

        // Extrapolate to 1M. No allocation — the sizes are arithmetic, and
    // building a real 1M index here would cost 700 MB to print four lines.
    size_t big_n = 1000000;
    size_t big_layer0 = big_n * (2 * M) * sizeof(int) + big_n * sizeof(int);
    size_t big_upper  = big_n * M       * sizeof(int) + big_n * sizeof(int);

    printf("\n    extrapolated to n=%zu, M=%d, 5 layers:\n", big_n, M);
    printf("    layer 0:      %.0f MB\n", big_layer0 / 1e6);
    printf("    layers 1-4:   %.0f MB each\n", big_upper / 1e6);
    printf("    graph total:  %.0f MB\n", (big_layer0 + 4 * big_upper) / 1e6);
    printf("    vectors:      %.0f MB\n", big_n * 128 * sizeof(float) / 1e6);

    hnsw_free(h); vs_free(vs); free(data);
}

int main(void) {
    printf("test_hnsw_storage:\n");
    test_create();
    test_ensure_level();
    test_ensure_level_idempotent();
    test_ensure_level_bounds();
    test_membership();
    test_layer_accessor();
    test_free();
    test_memory_report();
    printf("test_hnsw_storage: all passed\n");
    return 0;
}