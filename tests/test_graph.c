#include <assert.h>
#include <stdio.h>
#include "graph.h"
#include <stdlib.h>

// --- Case 1: a fresh graph has no edges -------------------------------
static void test_empty(void) {
    int n = 10, M = 4;
    Graph *g = graph_create(n, M);
    assert(g != NULL);

    assert(g->n == n);
    assert(g->M == M);

    for (int i = 0; i < g->n; i++) {
        assert(g->degrees[i] == 0);
    }
    int out_count;
    graph_neighbours(g, 1, &out_count);
    assert(out_count == 0);

    graph_neighbours(g, 2, &out_count);
    assert(out_count == 0);

    graph_neighbours(g, 3, &out_count);
    assert(out_count == 0);

    graph_free(g);
    printf("  empty: PASS\n");
}

// --- Case 2: edges come back in insertion order -----------------------
static void test_add_and_read(void) {
    Graph *g = graph_create(10, 4);

    graph_add_edge(g, 3, 7);
    graph_add_edge(g, 3, 1);
    graph_add_edge(g, 3, 9);

    assert(graph_degree(g, 3) == 3);

    int out_count;
    const int * ptr = graph_neighbours(g, 3, &out_count);
    assert(out_count == 3);
    assert(*ptr == 7);
    assert(*(ptr + 1) == 1);
    assert(*(ptr + 2) == 9);
    graph_free(g);
    printf("  add_and_read: PASS\n");
}

// --- Case 3: nodes don't interfere with each other --------------------
// THE important test. Catches a wrong stride in i*M+j — if the multiplier
// is wrong, writing node 1's edges will overwrite node 0's, or land
// outside the block entirely.
static void test_no_interference(void) {
    int n = 8, M = 4;
    Graph *g = graph_create(n, M);

    for (int i = 0; i < n; i++) {
        graph_add_edge(g, i, i* 10);
        graph_add_edge(g, i, i* 10 + 1);
        graph_add_edge(g, i, i* 10 + 2);
    }

    const int * ptr;
    for (int i = 0; i < n; i++) {
        int out_count;
        ptr = graph_neighbours(g, i, &out_count);
        assert(*ptr == i*10);
        assert(*(ptr+ 1) == i*10 + 1);
        assert(*(ptr+ 2) == i*10 + 2);
    }

    graph_free(g);
    printf("  no_interference: PASS\n");
}

// --- Case 4: a node at capacity refuses more edges --------------------
static void test_full(void) {
    int M = 4;
    Graph *g = graph_create(5, M);

    for (int i = 0; i < M; i++) {
        assert(graph_add_edge(g, 2, i) == 1);
    }

    assert(graph_add_edge(g, 2, 0) == 0);

    assert(graph_degree(g, 2) == M);

    int out_count;
    const int * ptr = graph_neighbours(g, 2, &out_count);
    for (int i = 0; i < M; i++) {
        assert(*(ptr + i) == i);
    }

    graph_free(g);
    printf("  full: PASS\n");
}

// --- Case 5: boundary nodes -------------------------------------------
// Node 0 and node n-1 are where off-by-one errors live. Node 0 also
// passes with a broken stride, since 0 * anything is 0 — so node n-1
// is the one that matters here.
static void test_boundaries(void) {
    int n = 6, M = 3;
    Graph *g = graph_create(n, M);

    for (int i = 0; i < M; i++) {
        graph_add_edge(g, 0, i);
    }

    int out_count;
    const int * ptr = graph_neighbours(g, 0, &out_count);
    for (int i = 0; i < M; i++) {
        assert(*(ptr + i) == i);
    }

    for (int i = 0; i < M; i++) {
        graph_add_edge(g, n-1, i*10);
    }

    ptr = graph_neighbours(g, n-1, &out_count);
    for (int i = 0; i < M; i++) {
        assert(*(ptr + i) == i*10);
    }

    ptr = graph_neighbours(g, 0, &out_count);
    for (int i = 0; i < M; i++) {
        assert(*(ptr + i) == i);
    }

    graph_free(g);
    printf("  boundaries: PASS\n");
}

// --- Case 6: M = 1, the degenerate case -------------------------------
static void test_m_one(void) {

    int n = 5;
    int M = 1;
    Graph * g = graph_create(n, M);

    for (int i = 0; i < n; i++) {
        graph_add_edge(g, i, (i+1)%n);
    }
    
    int out_count;
    const int * ptr = graph_neighbours(g, 0, &out_count);
    for (int i = 0; i < M; i++) {
        ptr = graph_neighbours(g, 0, &out_count);
        assert(*(ptr) == (i+1)%n);
    }
    assert(graph_add_edge(g, 0, 1) == 0);
    graph_free(g);


    printf("  m_one: PASS\n");
}

// --- Case 7: free(NULL) is safe ---------------------------------------
static void test_null_free(void) {
    graph_free(NULL);
    printf("  null_free: PASS\n");
}

// --- Case 8: same seed produces an identical graph --------------------
// Reproducibility is the whole reason the seed is a parameter. Without
// this, a recall change at step 18 can't be attributed to a code change.
static void test_random_deterministic(void) {
    int n = 100, M = 8;

    Graph *a = graph_create(n, M);
    Graph *b = graph_create(n, M);
    Graph *c = graph_create(n, M);

    graph_fill_random(a, 42);
    graph_fill_random(b, 42);
    graph_fill_random(c, 99);


    for (int i = 0; i < a->n; i++) {
        assert(a->degrees[i] == b->degrees[i]);
    }

    for (int i = 0; i < a->n * a->M; i++) {
        assert(a->neighbours[i] == b->neighbours[i]);
    }


    int differs = 0;
    for (int i = 0; i < n && !differs; i++) {
        int ca, cc;
        const int *na = graph_neighbours(a, i, &ca);
        const int *nc = graph_neighbours(c, i, &cc);

        if (ca != cc) { differs = 1; break; }
        for (int j = 0; j < ca; j++) {
            if (na[j] != nc[j]) { differs = 1; break; }
        }
    }
    assert(differs);

    graph_free(a); graph_free(b); graph_free(c);
    printf("  random_deterministic: PASS\n");
}

// --- Case 9: structural invariants ------------------------------------
// No self-loops, no duplicates, degrees in range. These are the three
// ways the retry loop can go wrong.
static void test_random_structure(void) {
    int n = 200, M = 8;
    Graph *g = graph_create(n, M);
    graph_fill_random(g, 7);

    for (int i = 0; i < n; i++) {
        int count;
        const int *nbrs = graph_neighbours(g, i, &count);


        assert(count >= 1);
        assert(count <= M);

        for (int j = 0; j < count; j++) {

            assert(nbrs[j] != i);

            assert(nbrs[j] >= 0);
            assert(nbrs[j] < n);

            for (int k = j+1; k < count; k++) {
                assert(nbrs[k] != nbrs[j]);
            }
        }
    }

    graph_free(g);
    printf("  random_structure: PASS\n");
}

// --- Case 10: M >= n is refused, not hung -----------------------------
// The retry loop can't find M distinct others when M >= n. It must be
// caught up front or the function spins forever.
static void test_random_impossible(void) {

    Graph * g = graph_create(5, 5);
    assert(graph_fill_random(g, 0) == 0);
    graph_free(g);

    g = graph_create(1, 5);
    assert(graph_fill_random(g, 0) == 0);
    graph_free(g);

    g = graph_create(5, 4);
    assert(graph_fill_random(g, 0) == 1);
    graph_free(g);
    printf("  random_impossible: PASS\n");
}

// --- Case 11: the graph is connected ----------------------------------
// If some nodes are unreachable from the entry point, search can never
// find them and recall at step 18 is capped for reasons unrelated to the
// search algorithm. Rules out a confusing failure mode before it happens.
static void test_random_connected(void) {
    int n = 500, M = 8;
    Graph *g = graph_create(n, M);
    graph_fill_random(g, 4);

    // Plain BFS from node 0, counting how many nodes are reachable.
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
            int next = nbrs[j];
            if (!visited[next]) {
                visited[next] = 1;
                queue[tail++] = next;   // safe: each node is pushed at most
                                        // once, so tail never exceeds n
            }
        }
    }

    int reached = 0;
    for (int i = 0; i < n; i++) reached += visited[i];

    // If this fails, print the damage before aborting — at step 21 you'll
    // want to know whether it's one stray node or half the graph.
    if (reached != n) {
        printf("    only %d of %d nodes reachable from node 0\n", reached, n);
    }
    assert(reached == n);

    free(visited);
    free(queue);
    graph_free(g);
    printf("  random_connected: PASS\n");
}

int main(void) {
    printf("test_graph:\n");
    test_empty();
    test_add_and_read();
    test_no_interference();
    test_full();
    test_boundaries();
    test_m_one();
    test_null_free();
    test_random_deterministic();
    test_random_impossible();
    test_random_structure();
    test_random_connected();
    printf("test_graph: all passed\n");
    return 0;
}