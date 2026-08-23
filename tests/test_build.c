#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "graph.h"
#include "vectors.h"
#include "visited.h"
#include "distance.h"

// --- Case 1: tiny graph, neighbours are verifiably the nearest --------
// 8 points on a line at 0,10,20,...,70 with M=2. Each node's two nearest
// are its immediate left and right, so the answer is checkable by hand.
static void test_line_neighbours(void) {
    int n = 8, M = 2;
    float data[] = {0,10,20,30,40,50,60,70};

    VectorStore *vs = vs_create(data, n, 1);
    Graph *g = graph_create(n, M);

    assert(graph_build(g, vs, 16) == 1);

    // Node 3 (at 30) should be connected to nodes 2 and 4 (at 20 and 40).
    // Not asserting exact membership — replacement and insertion order
    // make the graph slightly path-dependent. Assert the neighbours are
    // CLOSE, which is the property that matters.
    for (int i = 0; i < n; i++) {
        int count;
        const int *nbrs = graph_neighbours(g, i, &count);
        assert(count > 0);

        for (int j = 0; j < count; j++) {
            int gap = abs(nbrs[j] - i);
            // A proximity-built graph should not connect node 0 to node 7.
            assert(gap <= 3);
        }
    }

    graph_free(g); vs_free(vs);
    printf("  line_neighbours: PASS\n");
}

// --- Case 2: degrees are sane -----------------------------------------
static void test_degrees(void) {
    int n = 500, M = 8;
    float *data = malloc(n * sizeof(float));
    srand(1);
    for (int i = 0; i < n; i++) data[i] = (float)(rand() % 1000);

    VectorStore *vs = vs_create(data, n, 1);
    Graph *g = graph_create(n, M);
    graph_build(g, vs, 32);

    int zero_degree = 0, full = 0;
    for (int i = 0; i < n; i++) {
        int d = graph_degree(g, i);
        assert(d >= 0 && d <= M);
        if (d == 0) zero_degree++;
        if (d == M) full++;
    }
    printf("    degree 0: %d nodes, degree M: %d of %d\n",
           zero_degree, full, n);
    // Node 0 may legitimately have low degree — it was inserted first.
    assert(zero_degree <= 1);

    free(data); graph_free(g); vs_free(vs);
    printf("  degrees: PASS\n");
}

// --- Case 3: no self-loops, no duplicates -----------------------------
static void test_structure(void) {
    int n = 300, M = 8;
    float *data = malloc(n * sizeof(float));
    srand(2);
    for (int i = 0; i < n; i++) data[i] = (float)(rand() % 500);

    VectorStore *vs = vs_create(data, n, 1);
    Graph *g = graph_create(n, M);
    graph_build(g, vs, 32);

    for (int i = 0; i < n; i++) {
        int count;
        const int *nbrs = graph_neighbours(g, i, &count);
        for (int j = 0; j < count; j++) {
            assert(nbrs[j] != i);                 // no self-loop
            assert(nbrs[j] >= 0 && nbrs[j] < n);  // valid id
            for (int k = j + 1; k < count; k++)
                assert(nbrs[k] != nbrs[j]);       // no duplicate
        }
    }

    free(data); graph_free(g); vs_free(vs);
    printf("  structure: PASS\n");
}

// --- Case 4: CONNECTIVITY — expect this to be interesting -------------
// Same BFS as the random-graph test. A proximity-built graph is far more
// prone to fragmenting: nearby vectors connect to each other, and nothing
// forces a bridge between distant clusters. This is the failure that
// step 22's neighbour heuristic exists to prevent.
static void test_connectivity(void) {
    int n = 1000, M = 8;
    float *data = malloc(n * 2 * sizeof(float));

    // TWO tight clusters, deliberately far apart. This is the shape that
    // breaks naive construction — and it's what real embedding data
    // looks like, just less extreme.
    srand(3);
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
            if (!visited[nbrs[j]]) {
                visited[nbrs[j]] = 1;
                queue[tail++] = nbrs[j];
            }
        }
    }

    int reached = 0;
    for (int i = 0; i < n; i++) reached += visited[i];

    printf("    reachable from node 0: %d of %d (%.1f%%)\n",
           reached, n, 100.0 * reached / n);

    // NOT asserting full connectivity. Report it. If this comes back
    // near 50%, the two clusters are disconnected and search can never
    // cross between them — exactly the problem step 22 addresses.
    free(visited); free(queue); free(data);
    graph_free(g); vs_free(vs);
    printf("  connectivity: REPORTED (not asserted)\n");
}

// --- Case 5: search on the built graph beats random ---------------------
// The whole point. Same beam search, same ef, better graph.
static void test_beats_random(void) {
    int n = 1000, M = 8, ef = 20;
    float *data = malloc(n * sizeof(float));
    srand(4);

    MaxHeap * candidates = heap_create(n, 0);
    MaxHeap * results = heap_create(ef+1, 1);

    for (int i = 0; i < n; i++) data[i] = (float)(rand() % 10000);

    VectorStore *vs = vs_create(data, n, 1);
    VisitedSet *v = visited_create(n);

    Graph *rnd = graph_create(n, M);
    graph_fill_random(rnd, 4);

    Graph *built = graph_create(n, M);
    graph_build(built, vs, 64);

    int rnd_hits = 0, built_hits = 0;
    int   ids[1];
    float dists[1];
    int nd;

    for (int t = 0; t < 100; t++) {
        float query[] = {(float)(rand() % 10000)};

        // Exact answer by brute force.
        int   true_id = 0;
        float true_dist = l2sq_distance(vs_get(vs, 0), query, 1);
        for (int i = 1; i < n; i++) {
            float d = l2sq_distance(vs_get(vs, i), query, 1);
            if (d < true_dist) { true_dist = d; true_id = i; }
        }

        graph_beam_search(rnd, vs, query, 0, ef, 1, v, ids, dists, &nd, candidates, results);
        if (ids[0] == true_id) rnd_hits++;

        graph_beam_search(built, vs, query, 0, ef, 1, v, ids, dists, &nd, candidates, results);
        if (ids[0] == true_id) built_hits++;
    }

    printf("    random graph: %d/100    built graph: %d/100\n",
           rnd_hits, built_hits);
    assert(built_hits > rnd_hits);

    free(data); visited_free(v);
    graph_free(rnd); graph_free(built); vs_free(vs);
    heap_free(candidates);
    heap_free(results);
    printf("  beats_random: PASS\n");
}

int main(void) {
    printf("test_build:\n");
    test_line_neighbours();
    test_degrees();
    test_structure();
    test_connectivity();
    test_beats_random();
    printf("test_build: all passed\n");
    return 0;
}