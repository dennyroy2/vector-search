#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "graph.h"
#include "vectors.h"
#include "visited.h"

// --- Case 1: a path graph, walk from one end to the other -------------
// Nodes 0..9 sit at positions 0..9 on a line. Each connects only to its
// immediate left and right neighbours. Query at 9, entry at 0.
// Greedy must walk the whole chain and land on node 9.
static void test_path_graph(void) {
    int n = 10, M = 2;
    float data[] = {0,1,2,3,4,5,6,7,8,9};

    VectorStore *vs = vs_create(data, n, 1);
    Graph *g = graph_create(n, M);

    // Chain: i <-> i+1
    for (int i = 0; i < n - 1; i++) {
        graph_add_edge(g, i, i + 1);
        graph_add_edge(g, i + 1, i);
    }

    VisitedSet *v = visited_create(n);
    float query[] = {9.0f};
    float dist;
    int ndists;
    int hops;
    int found = graph_greedy_search(g, vs, query, 0, v, &dist, &ndists, &hops);

    assert(found == 9);
    assert(dist == 0.0f);
    // 9 hops, each examining at most 2 neighbours, plus the entry.
    // If ndists is much larger, the visited set isn't working.
    printf("    path: found=%d ndists=%d\n", found, ndists);
    assert(ndists <= 20);

    visited_free(v); graph_free(g); vs_free(vs);
    printf("  path_graph: PASS\n");
}

// --- Case 2: query sits exactly on the entry point --------------------
// Nothing can improve on distance 0, so it must stop immediately.
static void test_query_at_entry(void) {
    int n = 10, M = 2;
    float data[] = {0,1,2,3,4,5,6,7,8,9};

    VectorStore *vs = vs_create(data, n, 1);
    Graph *g = graph_create(n, M);
    for (int i = 0; i < n - 1; i++) {
        graph_add_edge(g, i, i + 1);
        graph_add_edge(g, i + 1, i);
    }

    VisitedSet *v = visited_create(n);
    float query[] = {0.0f};      // exactly node 0
    float dist;
    int ndists;
    int out_hops;
    int found = graph_greedy_search(g, vs, query, 0, v, &dist, &ndists, &out_hops);

    assert(found == 0);
    assert(dist == 0.0f);
    // Entry + its one neighbour. No hop taken.
    assert(ndists <= 3);

    visited_free(v); graph_free(g); vs_free(vs);
    printf("  query_at_entry: PASS\n");
}

// --- Case 3: fully connected graph — greedy must be exact -------------
// Every node sees every other, so the true nearest neighbour is visible
// from the entry point in one hop. If this fails, the comparison logic
// is wrong; there is nowhere for the search to get stuck.
static void test_fully_connected(void) {
    int n = 6;
    int M = n - 1;
    // Scattered 1-D positions so the answer isn't adjacent to the entry.
    float data[] = {0.0f, 50.0f, 12.0f, 31.0f, 7.0f, 44.0f};

    VectorStore *vs = vs_create(data, n, 1);
    Graph *g = graph_create(n, M);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) graph_add_edge(g, i, j);
        }
    }

    VisitedSet *v = visited_create(n);
    float dist;
    int ndists;

    // Query at 30 — nearest is node 3 (at 31), distance 1.
    float q1[] = {30.0f};
    int out_hops;
    int found = graph_greedy_search(g, vs, q1, 0, v, &dist, &ndists, &out_hops);
    assert(found == 3);
    assert(dist == 1.0f);

    // Query at 48 — nearest is node 5 (at 44), distance 16.
    float q2[] = {48.0f};
    found = graph_greedy_search(g, vs, q2, 0, v, &dist, &ndists, &out_hops);
    assert(found == 1);
    assert(dist == 4.0f);

    visited_free(v); graph_free(g); vs_free(vs);
    printf("  fully_connected: PASS\n");
}

// --- Case 4: greedy provably fails ------------------------------------
// TWO clusters, no bridge between them. Entry is in the wrong one.
// Greedy climbs to the best node it can reach and stops there — a LOCAL
// minimum. It cannot cross to the cluster holding the true answer.
//
// This test asserts the WRONG answer on purpose. It documents the exact
// limitation that step 19's beam search exists to address, and it will
// still pass afterwards, since beam search with ef=1 IS greedy search.
static void test_local_minimum(void) {
    int n = 6, M = 2;
    // Cluster A near 0: nodes 0,1,2.  Cluster B near 100: nodes 3,4,5.
    float data[] = {0.0f, 2.0f, 4.0f,  100.0f, 102.0f, 104.0f};

    VectorStore *vs = vs_create(data, n, 1);
    Graph *g = graph_create(n, M);

    // Edges WITHIN each cluster only. Nothing connects A to B.
    graph_add_edge(g, 0, 1); graph_add_edge(g, 1, 0);
    graph_add_edge(g, 1, 2); graph_add_edge(g, 2, 1);
    graph_add_edge(g, 3, 4); graph_add_edge(g, 4, 3);
    graph_add_edge(g, 4, 5); graph_add_edge(g, 5, 4);

    VisitedSet *v = visited_create(n);
    float dist;
    int ndists;

    // Query at 103 — the true nearest is node 4 (at 102), distance 1.
    // But we enter at node 0, in the disconnected cluster.
    float query[] = {103.0f};
    int out_hops;
    int found = graph_greedy_search(g, vs, query, 0, v, &dist, &ndists, &out_hops);

    // Greedy walks 0 -> 1 -> 2 and stops. Node 2 is the closest node
    // REACHABLE from the entry, but not the closest node overall.
    assert(found == 2);
    assert(found != 4);              // the true answer, unreachable
    printf("    local minimum: returned node %d (true answer is 4)\n", found);

    visited_free(v); graph_free(g); vs_free(vs);
    printf("  local_minimum: PASS (correctly fails)\n");
}

// --- Case 5: the visited set does its job -----------------------------
// A cycle would make an unguarded search revisit nodes forever. With
// stamping, each node is evaluated at most once per search, so the
// distance count is bounded by n.
static void test_visited_bounds_work(void) {
    int n = 8, M = 2;
    float data[] = {0,1,2,3,4,5,6,7};

    VectorStore *vs = vs_create(data, n, 1);
    Graph *g = graph_create(n, M);

    // A ring: every node connects to the next, wrapping around.
    for (int i = 0; i < n; i++) {
        graph_add_edge(g, i, (i + 1) % n);
        graph_add_edge(g, (i + 1) % n, i);
    }

    VisitedSet *v = visited_create(n);
    float query[] = {4.0f};
    float dist;
    int ndists;
    int out_hops;

    graph_greedy_search(g, vs, query, 0, v, &dist, &ndists, &out_hops);

    // Can never exceed n — every node evaluated at most once.
    assert(ndists <= n);

    visited_free(v); graph_free(g); vs_free(vs);
    printf("  visited_bounds_work: PASS\n");
}

// --- Case 6: consecutive searches don't leak state --------------------
// The generation counter is the whole reason the visited set is cheap.
// If reset were broken, the second search would see the first search's
// nodes as already visited and terminate immediately.
static void test_reset_between_searches(void) {
    int n = 10, M = 2;
    float data[] = {0,1,2,3,4,5,6,7,8,9};

    VectorStore *vs = vs_create(data, n, 1);
    Graph *g = graph_create(n, M);
    for (int i = 0; i < n - 1; i++) {
        graph_add_edge(g, i, i + 1);
        graph_add_edge(g, i + 1, i);
    }

    VisitedSet *v = visited_create(n);
    float dist;
    int ndists_first, ndists_second, out_hops;

    float query[] = {9.0f};

    int a = graph_greedy_search(g, vs, query, 0, v, &dist, &ndists_first, &out_hops);
    int b = graph_greedy_search(g, vs, query, 0, v, &dist, &ndists_second, &out_hops);

    // Identical inputs must give identical results AND identical work.
    assert(a == b);
    assert(ndists_first == ndists_second);

    visited_free(v); graph_free(g); vs_free(vs);
    printf("  reset_between_searches: PASS\n");
}

int main(void) {
    printf("test_greedy:\n");
    test_path_graph();
    test_query_at_entry();
    test_fully_connected();
    test_local_minimum();
    test_visited_bounds_work();
    test_reset_between_searches();
    printf("test_greedy: all passed\n");
    return 0;
}