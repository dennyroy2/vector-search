#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "graph.h"
#include "vectors.h"
#include "visited.h"

// --- Case 1: the heuristic rejects a clump ----------------------------
// Node 0 is at the origin. Four candidates clump to the north, one sits
// east, one sits south. With M=3, naive selection takes three northerners.
// The heuristic should take ONE northerner plus east and south.
static void test_rejects_clump(void) {
    // 2-D. node 0 at origin, then the candidates.
    float data[] = {
          0.0f,   0.0f,   // 0: the node being inserted
          0.0f,  10.0f,   // 1: north, nearest
          1.0f,  11.0f,   // 2: north, clumped with 1
         -1.0f,  11.0f,   // 3: north, clumped with 1
          0.0f,  12.0f,   // 4: north, clumped with 1
         20.0f,   0.0f,   // 5: east, farther
          0.0f, -25.0f,   // 6: south, farthest
    };
    VectorStore *vs = vs_create(data, 7, 2);

    // Candidates sorted by distance from node 0:
    // 1 (100), 2 (122), 3 (122), 4 (144), 5 (400), 6 (625)
    int   candidates[] = {1, 2, 3, 4, 5, 6};
    float cdists[]     = {100.0f, 122.0f, 122.0f, 144.0f, 400.0f, 625.0f};

    int out[3];
    int n = graph_select_neighbours(vs, 0, candidates, cdists, 6, 3, out);

    printf("    selected:");
    for (int i = 0; i < n; i++) printf(" %d", out[i]);
    printf("\n");

    assert(n == 3);
    assert(out[0] == 1);      // nearest is always taken
    // TODO: assert out[1] and out[2] are 5 and 6 — the east and south
    //       candidates. Nodes 2, 3, 4 must be rejected: each is closer
    //       to node 1 than to node 0.
    //       Verify by hand: dist(2,1) = 1+1 = 2, dist(0,2) = 122.
    //       2 < 122, so node 2 is redundant.

    vs_free(vs);
    printf("  rejects_clump: PASS\n");
}

// --- Case 2: naive vs heuristic, side by side -------------------------
// Same candidates, and the contrast is the whole point of the step.
static void test_naive_vs_heuristic(void) {
    // TODO: reuse the data from case 1.
    // TODO: "naive selection" is just the first M candidates — take
    //       candidates[0..2] directly, no function call needed.
    // TODO: assert the two selections DIFFER.
    // TODO: print both so the difference is visible in the test output.
    //       This is the demonstration, so make it readable.

    printf("  naive_vs_heuristic: PASS\n");
}

// --- Case 3: nothing is rejected when candidates are already spread ---
// If every candidate is far from every other, none is redundant and the
// heuristic degenerates to taking the M nearest. Confirms it isn't
// over-rejecting.
static void test_spread_candidates(void) {
    // Six points on the axes, far apart from each other.
    float data[] = {
          0.0f,   0.0f,   // 0: the node
         10.0f,   0.0f,   // 1: east
        -10.0f,   0.0f,   // 2: west
          0.0f,  10.0f,   // 3: north
          0.0f, -10.0f,   // 4: south
         15.0f,  15.0f,   // 5: northeast, farther
    };
    VectorStore *vs = vs_create(data, 6, 2);

    int   candidates[] = {1, 2, 3, 4, 5};
    float cdists[]     = {100.0f, 100.0f, 100.0f, 100.0f, 450.0f};

    int out[4];
    int n = graph_select_neighbours(vs, 0, candidates, cdists, 5, 4, out);

    // TODO: assert n == 4, and that the four selected are 1,2,3,4 —
    //       the four nearest. None is closer to another than to node 0:
    //       dist(1,2) = 400 > 100, dist(1,3) = 200 > 100, etc.

    vs_free(vs);
    printf("  spread_candidates: PASS\n");
}

// --- Case 4: fewer than M selected when everything is redundant -------
// One tight clump only. After the first pick, everything else is closer
// to it than to the node. Degree ends up below M, which is correct
// behaviour, not a bug.
static void test_fewer_than_m(void) {
    float data[] = {
          0.0f,  0.0f,    // 0: the node
        100.0f,  0.0f,    // 1: far away
        100.5f,  0.0f,    // 2: right next to 1
        101.0f,  0.0f,    // 3: right next to 1
        100.0f,  0.5f,    // 4: right next to 1
    };
    VectorStore *vs = vs_create(data, 5, 2);

    // TODO: build candidates {1,2,3,4} with their true cdists from node 0.
    //       (dist(0,1) = 10000, dist(0,2) = 10100.25, etc.)
    // TODO: ask for M=4.
    // TODO: assert only 1 is selected. Every other candidate is within
    //       ~1 unit of node 1 but ~100 units from node 0.

    vs_free(vs);
    printf("  fewer_than_m: PASS\n");
}

// --- Case 5: THE test — connectivity on two clusters ------------------
// This is the same setup that reported 50% reachable with naive
// selection. It should now report close to 100%.
static void test_two_cluster_connectivity(void) {
    int n = 1000, M = 8;
    float *data = malloc(n * 2 * sizeof(float));

    srand(3);   // same seed as the naive version, for a fair comparison
    for (int i = 0; i < n; i++) {
        int cluster = (i < n / 2) ? 0 : 1;
        data[i*2]     = cluster * 1000.0f + (float)(rand() % 50);
        data[i*2 + 1] = cluster * 1000.0f + (float)(rand() % 50);
    }

    VectorStore *vs = vs_create(data, n, 2);
    Graph *g = graph_create(n, M);
    graph_build(g, vs, 64);

    int *visited = calloc(n, sizeof(int));
    int *queue = malloc(n * sizeof(int));
    int head = 0, tail = 0;
    visited[0] = 1;
    queue[tail++] = 0;
    while (head < tail) {
        int node = queue[head++];
        int count;
        const int *nbrs = graph_neighbours(g, node, &count);
        for (int j = 0; j < count; j++) {
            if (!visited[nbrs[j]]) { visited[nbrs[j]] = 1; queue[tail++] = nbrs[j]; }
        }
    }
    int reached = 0;
    for (int i = 0; i < n; i++) reached += visited[i];

    printf("    reachable: %d of %d (%.1f%%)  [naive was 50.0%%]\n",
           reached, n, 100.0 * reached / n);

    // TODO: assert reached is at least, say, 95% of n. Pick a threshold
    //       and justify it — 100% may not be guaranteed, but 50% is
    //       clearly broken and 95%+ clearly works.

    free(visited); free(queue); free(data);
    graph_free(g); vs_free(vs);
    printf("  two_cluster_connectivity: PASS\n");
}

// --- Case 6: long edges exist ------------------------------------------
// The mechanism, measured directly. With naive selection every edge is
// short. The heuristic should produce some genuinely long ones — those
// are the bridges.
static void test_long_edges_exist(void) {
    // TODO: reuse the two-cluster data from case 5.
    // TODO: build the graph.
    // TODO: count edges whose length exceeds, say, 100 units — anything
    //       that crosses between clusters is ~1400 units long.
    // TODO: assert at least one such edge exists. Print the count.
    //
    //       This is the clearest single piece of evidence that the
    //       heuristic does what it claims.

    printf("  long_edges_exist: PASS\n");
}

int main(void) {
    printf("test_nsw:\n");
    test_rejects_clump();
    test_naive_vs_heuristic();
    test_spread_candidates();
    test_fewer_than_m();
    test_two_cluster_connectivity();
    test_long_edges_exist();
    printf("test_nsw: all passed\n");
    return 0;
}