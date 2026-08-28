#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hnsw.h"
#include "vectors.h"

#define TEST_PATH "build/test_index.bin"

static HNSW *build_test_index(VectorStore *vs, int M) {
    HNSW *h = hnsw_create(vs, M);
    assert(h != NULL);
    assert(hnsw_build(h, 64, 42) == 1);
    return h;
}

static float *make_data(int n, int dim, unsigned seed) {
    float *d = malloc((size_t)n * dim * sizeof(float));
    srand(seed);
    for (int i = 0; i < n * dim; i++) d[i] = (float)(rand() % 1000);
    return d;
}

// --- Case 1: round-trip identity --------------------------------------
// THE test. Every field and every array must survive unchanged.
static void test_roundtrip(void) {
    int n = 2000, dim = 8, M = 8;
    float *data = make_data(n, dim, 1);
    VectorStore *vs = vs_create(data, n, dim);

    HNSW *orig = build_test_index(vs, M);
    assert(hnsw_save(orig, TEST_PATH) == 1);

    HNSW *load = hnsw_load(TEST_PATH, vs);
    assert(load != NULL);

    assert(load->n == orig->n);
    assert(load->M == orig->M);
    assert(load->M0 == orig->M0);
    assert(load->max_level == orig->max_level);
    assert(load->entry_point == orig->entry_point);
    assert(load->mL == orig->mL);

    for (int i = 0; i < n; i++)
        assert(load->node_levels[i] == orig->node_levels[i]);

    for (int l = 0; l <= orig->max_level; l++) {
        Graph *go = hnsw_layer(orig, l);
        Graph *gl = hnsw_layer(load, l);
        assert(gl != NULL);
        assert(gl->M == go->M);       // layer 0 must come back with M0

        for (int i = 0; i < n; i++) {
            assert(gl->degrees[i] == go->degrees[i]);
            for (int j = 0; j < go->degrees[i]; j++) {
                assert(gl->neighbours[i * go->M + j] ==
                       go->neighbours[i * go->M + j]);
            }
        }
    }

    hnsw_free(orig); hnsw_free(load); vs_free(vs); free(data);
    printf("  roundtrip: PASS\n");
}

// --- Case 2: searches return identical results ------------------------
// Structural equality should imply this, but this is the property you
// actually care about.
static void test_search_matches(void) {
    int n = 2000, dim = 8, M = 8;
    float *data = make_data(n, dim, 2);
    VectorStore *vs = vs_create(data, n, dim);

    HNSW *orig = build_test_index(vs, M);
    assert(hnsw_save(orig, TEST_PATH) == 1);
    HNSW *load = hnsw_load(TEST_PATH, vs);
    assert(load != NULL);

    VisitedSet *v = visited_create(n);
    MaxHeap *cand = heap_create(n, 0);
    MaxHeap *res  = heap_create(101, 1);

    int   ids_a[10], ids_b[10];
    float d_a[10], d_b[10];
    int   nd_a, nd_b, desc_a, desc_b;

    for (int q = 0; q < 50; q++) {
        const float *query = vs_get(vs, q * 37 % n);

        int ca = hnsw_search(orig, query, 10, 50, v, cand, res,
                             ids_a, d_a, &nd_a, &desc_a);
        int cb = hnsw_search(load, query, 10, 50, v, cand, res,
                             ids_b, d_b, &nd_b, &desc_b);

        assert(ca == cb);
        assert(nd_a == nd_b);          // same work, not just same answer
        for (int i = 0; i < ca; i++) {
            assert(ids_a[i] == ids_b[i]);
            assert(d_a[i] == d_b[i]);
        }
    }

    visited_free(v); heap_free(cand); heap_free(res);
    hnsw_free(orig); hnsw_free(load); vs_free(vs); free(data);
    printf("  search_matches: PASS\n");
}

// --- Case 3: file size is exactly what the format predicts ------------
// Catches a missing write, a duplicated one, or a wrong-width layer.
static void test_file_size(void) {
    int n = 1000, dim = 4, M = 8;
    float *data = make_data(n, dim, 3);
    VectorStore *vs = vs_create(data, n, dim);
    HNSW *h = build_test_index(vs, M);
    assert(hnsw_save(h, TEST_PATH) == 1);

    // header: 8 ints + 1 double
    size_t expected = 8 * sizeof(int) + sizeof(double);
    // node_levels
    expected += (size_t)n * sizeof(int);
    // each layer: degrees (n ints) + neighbours (n * that layer's M ints)
    for (int l = 0; l <= h->max_level; l++) {
        int width = hnsw_layer(h, l)->M;
        expected += (size_t)n * sizeof(int);
        expected += (size_t)n * width * sizeof(int);
    }

    FILE *f = fopen(TEST_PATH, "rb");
    fseek(f, 0, SEEK_END);
    long actual = ftell(f);
    fclose(f);

    printf("    expected %zu bytes, got %ld\n", expected, actual);
    assert((size_t)actual == expected);

    hnsw_free(h); vs_free(vs); free(data);
    printf("  file_size: PASS\n");
}

// --- Case 4: a file that isn't ours is rejected -----------------------
// Must fail on the magic BEFORE allocating, or a bogus n gets passed
// to malloc.
static void test_bad_magic(void) {
    FILE *f = fopen(TEST_PATH, "wb");
    int junk[64];
    for (int i = 0; i < 64; i++) junk[i] = 0x41414141;
    fwrite(junk, sizeof(int), 64, f);
    fclose(f);

    int n = 100, dim = 4;
    float *data = make_data(n, dim, 4);
    VectorStore *vs = vs_create(data, n, dim);

    assert(hnsw_load(TEST_PATH, vs) == NULL);

    vs_free(vs); free(data);
    printf("  bad_magic: PASS\n");
}

// --- Case 5: version mismatch is rejected -----------------------------
static void test_bad_version(void) {
    int n = 500, dim = 4, M = 8;
    float *data = make_data(n, dim, 5);
    VectorStore *vs = vs_create(data, n, dim);
    HNSW *h = build_test_index(vs, M);
    assert(hnsw_save(h, TEST_PATH) == 1);
    hnsw_free(h);

    // Overwrite the version field (second int) in place.
    FILE *f = fopen(TEST_PATH, "r+b");
    int bad = HNSW_VERSION + 99;
    fseek(f, sizeof(int), SEEK_SET);
    fwrite(&bad, sizeof(int), 1, f);
    fclose(f);

    assert(hnsw_load(TEST_PATH, vs) == NULL);

    vs_free(vs); free(data);
    printf("  bad_version: PASS\n");
}

// --- Case 6: mismatched vector store is rejected ----------------------
// The index stores node IDs, which index into the store. Pairing it with
// the wrong store gives silently wrong results, so n and dim are checked.
static void test_wrong_store(void) {
    int n = 1000, dim = 4, M = 8;
    float *data = make_data(n, dim, 6);
    VectorStore *vs = vs_create(data, n, dim);
    HNSW *h = build_test_index(vs, M);
    assert(hnsw_save(h, TEST_PATH) == 1);
    hnsw_free(h);

    // Wrong count.
    float *small_data = make_data(500, dim, 7);
    VectorStore *small = vs_create(small_data, 500, dim);
    assert(hnsw_load(TEST_PATH, small) == NULL);

    // Wrong dimension.
    float *wide_data = make_data(n, 16, 8);
    VectorStore *wide = vs_create(wide_data, n, 16);
    assert(hnsw_load(TEST_PATH, wide) == NULL);

    vs_free(small); free(small_data);
    vs_free(wide);  free(wide_data);
    vs_free(vs);    free(data);
    printf("  wrong_store: PASS\n");
}

// --- Case 7: a truncated file fails rather than crashing --------------
static void test_truncated(void) {
    int n = 1000, dim = 4, M = 8;
    float *data = make_data(n, dim, 9);
    VectorStore *vs = vs_create(data, n, dim);
    HNSW *h = build_test_index(vs, M);
    assert(hnsw_save(h, TEST_PATH) == 1);
    hnsw_free(h);

    // Read the whole file, write back only the first half.
    FILE *f = fopen(TEST_PATH, "rb");
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(size);
    fread(buf, 1, size, f);
    fclose(f);

    f = fopen(TEST_PATH, "wb");
    fwrite(buf, 1, size / 2, f);
    fclose(f);
    free(buf);

    // Header survives, so the magic and version checks pass — the failure
    // must be caught by a short fread on one of the arrays.
    assert(hnsw_load(TEST_PATH, vs) == NULL);

    vs_free(vs); free(data);
    printf("  truncated: PASS\n");
}

// --- Case 8: missing file --------------------------------------------
static void test_missing_file(void) {
    int n = 100, dim = 4;
    float *data = make_data(n, dim, 10);
    VectorStore *vs = vs_create(data, n, dim);

    assert(hnsw_load("build/does_not_exist_xyz.bin", vs) == NULL);

    vs_free(vs); free(data);
    printf("  missing_file: PASS\n");
}

int main(void) {
    printf("test_hnsw_io:\n");
    test_roundtrip();
    test_search_matches();
    test_file_size();
    test_bad_magic();
    test_bad_version();
    test_wrong_store();
    test_truncated();
    test_missing_file();
    remove(TEST_PATH);
    printf("test_hnsw_io: all passed\n");
    return 0;
}