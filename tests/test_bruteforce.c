#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "bruteforce.h"
#include "vectors.h"

static int close(float got, float expected) {
    return fabsf(got - expected) < 1e-5f;
}

// --- Case 1: points on a line -----------------------------------------
// 1-D vectors at 0,1,2,...,9. Query at 0.
// Nearest are ids 0,1,2 at squared distances 0,1,4.
static void test_line(void) {
    float data[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    VectorStore *vs = vs_create(data, 10, 1);
    assert(vs != NULL);

    float query[] = {0.0f};
    int   ids[3];
    float dists[3];

    int count = bruteforce_topk(vs, query, 3, ids, dists);

    assert(count == 3);
    assert(ids[0] == 0 && ids[1] == 1 && ids[2] == 2);
    assert(close(dists[0], 0.0f));
    assert(close(dists[1], 1.0f));
    assert(close(dists[2], 4.0f));

    vs_free(vs);
    printf("  line: PASS\n");
}

// --- Case 2: ascending distance order ---------------------------------
// Scattered positions so a backwards-drain bug is visible.
static void test_ordering(void) {
    float data[] = {17.0f, 3.0f, 42.0f, 8.0f, 25.0f, 1.0f, 60.0f, 12.0f};
    VectorStore *vs = vs_create(data, 8, 1);

    float query[] = {10.0f};
    int   ids[5];
    float dists[5];

    int count = bruteforce_topk(vs, query, 5, ids, dists);
    assert(count == 5);

    for (int i = 0; i < count - 1; i++) {
        assert(dists[i] <= dists[i + 1]);
    }

    // Closest to 10 is 8 (id 3, dist 4), then 12 (id 7, dist 4)...
    // 8 and 12 are both distance 4 — a tie. Just check the nearest
    // distance is 4, not which id won it.
    assert(close(dists[0], 4.0f));

    vs_free(vs);
    printf("  ordering: PASS\n");
}

// --- Case 3: k larger than the store ----------------------------------
// Must return n, not crash, not write past slot n-1.
static void test_k_exceeds_n(void) {
    float data[] = {0.0f, 10.0f, 20.0f, 30.0f, 40.0f};
    VectorStore *vs = vs_create(data, 5, 1);

    float query[] = {0.0f};

    // Poison the tail so an over-write is detectable.
    int   ids[20];
    float dists[20];
    for (int i = 0; i < 20; i++) { ids[i] = -999; dists[i] = -999.0f; }

    int count = bruteforce_topk(vs, query, 20, ids, dists);

    assert(count == 5);
    for (int i = 0; i < count - 1; i++) assert(dists[i] <= dists[i + 1]);
    assert(ids[0] == 0 && close(dists[0], 0.0f));

    // Slots beyond count must be untouched.
    for (int i = count; i < 20; i++) {
        assert(ids[i] == -999);
        assert(dists[i] == -999.0f);
    }

    vs_free(vs);
    printf("  k_exceeds_n: PASS\n");
}

// --- Case 4: k=1 must agree with bruteforce_nn ------------------------
// Two independent paths — bruteforce_nn has no heap in it at all.
static void test_agrees_with_nn(void) {
    // 12 vectors in 2-D, scattered.
    float data[] = {
         0.0f,  0.0f,   3.0f,  4.0f,  -2.0f,  1.0f,   7.0f, -3.0f,
         1.5f,  1.5f,  -5.0f, -5.0f,   9.0f,  9.0f,   0.5f, -0.5f,
        -3.0f,  6.0f,   4.0f,  0.0f,   2.0f, -7.0f,  -8.0f,  2.0f,
    };
    VectorStore *vs = vs_create(data, 12, 2);

    float queries[] = {
         0.0f,  0.0f,
         5.0f,  5.0f,
        -4.0f, -4.0f,
         1.0f, -1.0f,
         8.0f,  8.0f,
    };

    for (int q = 0; q < 5; q++) {
        const float *query = queries + q * 2;

        int   nn_id;
        float nn_dist;
        bruteforce_nn(vs, query, &nn_id, &nn_dist);

        int   ids[1];
        float dists[1];
        int count = bruteforce_topk(vs, query, 1, ids, dists);

        assert(count == 1);
        assert(ids[0] == nn_id);
        assert(close(dists[0], nn_dist));
    }

    vs_free(vs);
    printf("  agrees_with_nn: PASS\n");
}

// --- Case 5: k equals n -----------------------------------------------
// Everything returned, fully sorted, every id exactly once.
static void test_k_equals_n(void) {
    int n = 8;
    float data[] = {5.0f, 2.0f, 9.0f, 1.0f, 7.0f, 3.0f, 8.0f, 4.0f};
    VectorStore *vs = vs_create(data, n, 1);

    float query[] = {0.0f};
    int   ids[8];
    float dists[8];

    int count = bruteforce_topk(vs, query, n, ids, dists);
    assert(count == n);

    for (int i = 0; i < count - 1; i++) assert(dists[i] <= dists[i + 1]);

    // Every id must appear exactly once — catches a drain that pops
    // the same slot twice or skips one.
    int seen[8] = {0};
    for (int i = 0; i < count; i++) {
        assert(ids[i] >= 0 && ids[i] < n);
        seen[ids[i]]++;
    }
    for (int i = 0; i < n; i++) assert(seen[i] == 1);

    vs_free(vs);
    printf("  k_equals_n: PASS\n");
}

// --- Case 6: query is itself in the store -----------------------------
static void test_query_in_store(void) {
    float data[] = {
        1.0f, 2.0f,   30.0f, 40.0f,   -5.0f, -6.0f,   7.0f, 8.0f,
    };
    VectorStore *vs = vs_create(data, 4, 2);

    for (int i = 0; i < 4; i++) {
        const float *query = data + i * 2;
        int   ids[2];
        float dists[2];

        int count = bruteforce_topk(vs, query, 2, ids, dists);
        assert(count == 2);
        assert(ids[0] == i);
        assert(dists[0] == 0.0f);      // exact — no arithmetic error possible
    }

    vs_free(vs);
    printf("  query_in_store: PASS\n");
}

// --- Case 7: degenerate inputs ----------------------------------------
static void test_degenerate(void) {
    float data[] = {1.0f, 2.0f, 3.0f};
    VectorStore *vs = vs_create(data, 3, 1);
    float query[] = {0.0f};

    int   ids[3]   = {-999, -999, -999};
    float dists[3] = {-999.0f, -999.0f, -999.0f};

    // k = 0 and k < 0 must return 0 and write nothing.
    assert(bruteforce_topk(vs, query, 0, ids, dists) == 0);
    assert(bruteforce_topk(vs, query, -5, ids, dists) == 0);
    assert(ids[0] == -999 && dists[0] == -999.0f);

    vs_free(vs);

    // A single-vector store.
    float one[] = {42.0f};
    VectorStore *vs1 = vs_create(one, 1, 1);
    int   id1[1];
    float d1[1];
    assert(bruteforce_topk(vs1, query, 1, id1, d1) == 1);
    assert(id1[0] == 0);
    assert(close(d1[0], 42.0f * 42.0f));
    vs_free(vs1);

    printf("  degenerate: PASS\n");
}

int main(void) {
    printf("test_bruteforce:\n");
    test_line();
    test_ordering();
    test_k_exceeds_n();
    test_agrees_with_nn();
    test_k_equals_n();
    test_query_in_store();
    test_degenerate();
    printf("test_bruteforce: all passed\n");
    return 0;
}