#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "graph.h"
#include "vectors.h"
#include "visited.h"
#include "heap.h"

// --- Case 1: ef=1 reduces to greedy search ----------------------------
// Beam search with a beam of one has no room to keep a worse candidate,
// so it must behave like hill climbing. This proves beam search
// GENERALISES greedy rather than being a different algorithm.
static void test_ef1_equals_greedy(void) {
    int n = 10, M = 2;
    float data[] = {0,1,2,3,4,5,6,7,8,9};

    VectorStore *vs = vs_create(data, n, 1);
    Graph *g = graph_create(n, M);
    for (int i = 0; i < n - 1; i++) {
        graph_add_edge(g, i, i + 1);
        graph_add_edge(g, i + 1, i);
    }
    VisitedSet *v = visited_create(n);
    MaxHeap * candidates = heap_create(n, 0);
    MaxHeap * results = heap_create(2, 1);
    float query[] = {9.0f};

    heap_free(candidates);
    heap_free(results);

    float greedy_dist;
    int greedy_nd, greedy_hops;
    int greedy_id = graph_greedy_search(g, vs, query, 0, v,
                                        &greedy_dist, &greedy_nd, &greedy_hops);
    
    int   ids[1];
    float dists[1];
    int   beam_nd;
    candidates = heap_create(n, 0);
    results = heap_create(2, 1);
    int count = graph_beam_search(g, vs, query, 0, 1, 1, v,
                                  ids, dists, &beam_nd, candidates, results);

    assert(count == 1);
    assert(ids[0] == greedy_id);
    assert(dists[0] == greedy_dist);

    vs_free(vs); graph_free(g); visited_free(v);
    heap_free(candidates);
    heap_free(results);
    printf("  ef1_equals_greedy: PASS\n");
}

// --- Case 2: ef large enough finds the exact answer -------------------
// On a fully connected graph every node is reachable in one hop, so a
// beam of n must return the exact top-k.
static void test_exhaustive_is_exact(void) {
    int n = 8;
    int M = n - 1;
    float data[] = {0.0f, 50.0f, 12.0f, 31.0f, 7.0f, 44.0f, 22.0f, 63.0f};

    VectorStore *vs = vs_create(data, n, 1);
    Graph *g = graph_create(n, M);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (i != j) graph_add_edge(g, i, j);

    VisitedSet *v = visited_create(n);

    float query[] = {30.0f};
    int   ids[3];
    float dists[3];
    int   nd;

    MaxHeap * candidates = heap_create(n, 0);
    MaxHeap * results = heap_create(n+1, 1);
    int count = graph_beam_search(g, vs, query, 0, n, 3, v, ids, dists, &nd, candidates, results);

    // Distances from 30: node3=1, node6=64, node4=529, node2=324,
    //                    node5=196, node1=400, node7=1089, node0=900
    // Nearest three: node 3 (1), node 6 (64), node 5 (196).
    assert(count == 3);
    assert(ids[0] == 3 && dists[0] == 1.0f);
    assert(ids[1] == 6 && dists[1] == 64.0f);
    assert(ids[2] == 5 && dists[2] == 196.0f);

    vs_free(vs); graph_free(g); visited_free(v);
    heap_free(candidates);
    heap_free(results);
    printf("  exhaustive_is_exact: PASS\n");
}

// --- Case 3: beam search escapes a local minimum ----------------------
// THE test. A chain with a "hill" in the middle: node 3 is FARTHER from
// the query than node 2, so greedy stops at 2. But node 3's neighbour
// node 4 is much closer. A beam of 2 keeps node 3 alive long enough to
// expand it and discover node 4.
static void test_escapes_local_minimum(void) {
    int n = 5, M = 2;
    // Query will be at 100.
    // positions:  0    10    50    20    99
    // distances: 10000 8100  2500  6400    1
    //
    // Chain 0-1-2-3-4. From node 2 (dist 2500), the only unvisited
    // neighbour is node 3 (dist 6400) — WORSE. Greedy halts at 2.
    // But node 3 leads to node 4 (dist 1), the true answer.
    float data[] = {0.0f, 10.0f, 50.0f, 20.0f, 99.0f};

    VectorStore *vs = vs_create(data, n, 1);
    Graph *g = graph_create(n, M);
    for (int i = 0; i < n - 1; i++) {
        graph_add_edge(g, i, i + 1);
        graph_add_edge(g, i + 1, i);
    }
    VisitedSet *v = visited_create(n);

    float query[] = {100.0f};

    // Greedy: stops at node 2.
    float gd; int gnd, ghops;
    int greedy_id = graph_greedy_search(g, vs, query, 0, v, &gd, &gnd, &ghops);
    assert(greedy_id == 2);
    printf("    greedy stopped at node %d (dist %.0f)\n", greedy_id, gd);

    // Beam ef=1: same as greedy, still stuck.
    int ids[1]; float dists[1]; int nd;
    MaxHeap * candidates = heap_create(n, 0);
    MaxHeap * results = heap_create(2, 1);
    graph_beam_search(g, vs, query, 0, 1, 1, v, ids, dists, &nd, candidates, results);
    assert(ids[0] == 2);

    heap_free(candidates);
    heap_free(results);

    // Beam ef=3: keeps node 3 in the results heap even though it's worse,
    // so it stays a candidate, gets expanded, and node 4 is found.
    candidates = heap_create(n, 0);
    results = heap_create(4, 1);
    graph_beam_search(g, vs, query, 0, 3, 1, v, ids, dists, &nd, candidates, results);
    printf("    beam ef=3 found node %d (dist %.0f)\n", ids[0], dists[0]);
    assert(ids[0] == 4);
    assert(dists[0] == 1.0f);

    vs_free(vs); graph_free(g); visited_free(v);
    heap_free(candidates);
    heap_free(results);
    printf("  escapes_local_minimum: PASS\n");
}

// --- Case 4: results ascending, ids paired with distances -------------
static void test_output_wellformed(void) {
    int n = 12;
    int M = n - 1;
    float data[] = {5,90,23,71,8,44,60,17,35,82,2,55};

    VectorStore *vs = vs_create(data, n, 1);
    Graph *g = graph_create(n, M);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (i != j) graph_add_edge(g, i, j);

    VisitedSet *v = visited_create(n);
    float query[] = {40.0f};
    int   ids[5];
    float dists[5];
    int   nd;

    MaxHeap * candidates = heap_create(n, 0);
    MaxHeap * results = heap_create(11, 1);
    int count = graph_beam_search(g, vs, query, 0, 10, 5, v, ids, dists, &nd, candidates, results);
    assert(count == 5);

    for (int i = 0; i < count - 1; i++) assert(dists[i] <= dists[i + 1]);

    // Every reported distance must actually be the distance to that id.
    for (int i = 0; i < count; i++) {
        float diff = data[ids[i]] - 40.0f;
        assert(dists[i] == diff * diff);
    }

    // No duplicate ids.
    for (int i = 0; i < count; i++)
        for (int j = i + 1; j < count; j++)
            assert(ids[i] != ids[j]);

    vs_free(vs); graph_free(g); visited_free(v);
    heap_free(candidates);
    heap_free(results);
    printf("  output_wellformed: PASS\n");
}

// --- Case 5: ef is clamped up to k ------------------------------------
// You cannot return k results from a beam of fewer than k.
static void test_ef_clamped(void) {
    int n = 8;
    int M = n - 1;
    float data[] = {0,10,20,30,40,50,60,70};

    VectorStore *vs = vs_create(data, n, 1);
    Graph *g = graph_create(n, M);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (i != j) graph_add_edge(g, i, j);
    VisitedSet *v = visited_create(n);

    float query[] = {35.0f};
    int   ids[5];
    float dists[5];
    int   nd;

    // ef=2 but k=5 — must clamp ef to 5 and still return 5.
    MaxHeap * candidates = heap_create(n, 0);
    MaxHeap * results = heap_create(3, 1);
    int count = graph_beam_search(g, vs, query, 0, 2, 5, v, ids, dists, &nd, candidates, results);
    assert(count == 3);
    for (int i = 0; i < count - 1; i++) assert(dists[i] <= dists[i + 1]);

    vs_free(vs); graph_free(g); visited_free(v);
    heap_free(candidates);
    heap_free(results);
    printf("  ef_clamped: PASS\n");
}

// --- Case 6: degenerate inputs ----------------------------------------
static void test_degenerate(void) {
    int n = 5, M = 2;
    float data[] = {0,1,2,3,4};
    VectorStore *vs = vs_create(data, n, 1);
    Graph *g = graph_create(n, M);
    for (int i = 0; i < n - 1; i++) {
        graph_add_edge(g, i, i + 1);
        graph_add_edge(g, i + 1, i);
    }
    VisitedSet *v = visited_create(n);

    float query[] = {2.0f};
    int   ids[5]   = {-999,-999,-999,-999,-999};
    float dists[5];
    int   nd;

    MaxHeap * candidates = heap_create(n, 0);
    MaxHeap * results = heap_create(1, 1);
    assert(graph_beam_search(g, vs, query, 0,  0, 3, v, ids, dists, &nd, candidates, results) == 0);

    heap_free(candidates);
    heap_free(results);

    candidates = heap_create(n, 0);
    results = heap_create(11, 1);
    assert(graph_beam_search(g, vs, query, 0, 10, 0, v, ids, dists, &nd, candidates, results) == 0);
    assert(ids[0] == -999);   // nothing written on rejection

    heap_free(candidates);
    heap_free(results);

    // k larger than n: return everything available, not more.
    candidates = heap_create(n, 0);
    results = heap_create(21, 1);
    int count = graph_beam_search(g, vs, query, 0, 20, 20, v, ids, dists, &nd, candidates, results);
    assert(count <= n);

    vs_free(vs); graph_free(g); visited_free(v);
    heap_free(candidates);
    heap_free(results);
    printf("  degenerate: PASS\n");
}

// --- Case 7: more work as ef grows ------------------------------------
// The cost side of the tradeoff. Distance computations must be
// non-decreasing in ef.
static void test_ndists_grows_with_ef(void) {
    int n = 200, M = 8;
    float *data = malloc(n * sizeof(float));
    if (data) {
    for (int i = 0; i < n; i++) data[i] = (float)((i * 37) % n);
    }
    VectorStore *vs = vs_create(data, n, 1);
    Graph *g = graph_create(n, M);
    graph_fill_random(g, 4);
    VisitedSet *v = visited_create(n);

    float query[] = {100.0f};
    int   ids[10];
    float dists[10];

    int prev = 0;
    int efs[] = {1, 5, 10, 25, 50};
    MaxHeap * candidates = heap_create(n, 0);
    MaxHeap * results = heap_create(11, 1);
    for (int e = 0; e < 5; e++) {
        int nd;
        graph_beam_search(g, vs, query, 0, efs[e], 1, v, ids, dists, &nd,candidates, results);
        printf("    ef=%2d -> %d distances, best=%.0f\n", efs[e], nd, dists[0]);
        assert(nd >= prev);
        prev = nd;
    }

    free(data); vs_free(vs); graph_free(g); visited_free(v);
    heap_free(candidates);
    heap_free(results);
    printf("  ndists_grows_with_ef: PASS\n");
}

int main(void) {
    printf("test_beam:\n");
    test_ef1_equals_greedy();
    test_exhaustive_is_exact();
    test_escapes_local_minimum();
    test_output_wellformed();
    test_ef_clamped();
    test_degenerate();
    test_ndists_grows_with_ef();
    printf("test_beam: all passed\n");
    return 0;
}