#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "heap.h"

// --- Case 1: pop order is descending by distance ----------------------
// The core property of a max-heap. This one test catches most sift bugs.
static void test_pop_order(void) {
    MaxHeap *h = heap_create(20);
    assert(h != NULL);

    float dists[] = {5.0f, 1.0f, 9.0f, 3.0f, 7.0f, 2.0f, 8.0f, 4.0f, 6.0f};
    int n = 9;

    for (int i = 0; i < n; i++) {
        assert(heap_push(h, i, dists[i]) == 1);
    }
    assert(heap_size(h) == n);

    Candidate prev, cur;
    assert(heap_pop(h, &prev) == 1);
    for (int i = 1; i < n; i++) {
        assert(heap_pop(h, &cur) == 1);
        assert(cur.dist <= prev.dist);   // non-increasing
        prev = cur;
    }
    assert(heap_size(h) == 0);

    heap_free(h);
    printf("  pop_order: PASS\n");
}

// --- Case 2: insertion order must not matter --------------------------
// Ascending, descending, and shuffled input all exercise different sift
// paths but must produce identical output.
static void test_insertion_orders(void) {
    float ascending[]  = {1, 2, 3, 4, 5, 6, 7, 8};
    float descending[] = {8, 7, 6, 5, 4, 3, 2, 1};
    float shuffled[]   = {4, 8, 1, 6, 3, 7, 2, 5};
    int n = 8;

    float *orders[] = {ascending, descending, shuffled};

    for (int o = 0; o < 3; o++) {
        MaxHeap *h = heap_create(n);
        for (int i = 0; i < n; i++) heap_push(h, i, orders[o][i]);

        // Whatever went in, 8.0 comes out first and 1.0 last.
        for (int expected = n; expected >= 1; expected--) {
            Candidate c;
            assert(heap_pop(h, &c) == 1);
            assert(c.dist == (float)expected);
        }
        heap_free(h);
    }
    printf("  insertion_orders: PASS\n");
}

// --- Case 3: the id must travel with the distance ---------------------
// A swap that moves .dist but not .id passes every ordering test above
// and is completely broken. The ids ARE the search result.
static void test_id_pairing(void) {
    MaxHeap *h = heap_create(10);

    // Deliberately make id and dist unrelated, so a mixup is visible.
    // id 100 -> dist 3.0, id 200 -> dist 1.0, etc.
    int   ids[]   = {100, 200, 300, 400, 500};
    float dists[] = {3.0f, 1.0f, 5.0f, 2.0f, 4.0f};

    for (int i = 0; i < 5; i++) heap_push(h, ids[i], dists[i]);

    // Expected pop order by distance: 5.0(300), 4.0(500), 3.0(100),
    //                                 2.0(400), 1.0(200)
    int   expect_ids[]   = {300, 500, 100, 400, 200};
    float expect_dists[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    for (int i = 0; i < 5; i++) {
        Candidate c;
        assert(heap_pop(h, &c) == 1);
        assert(c.dist == expect_dists[i]);
        assert(c.id   == expect_ids[i]);   // the assertion that matters
    }

    heap_free(h);
    printf("  id_pairing: PASS\n");
}

// --- Case 4: peek agrees with the next pop ----------------------------
static void test_peek(void) {
    MaxHeap *h = heap_create(10);
    float dists[] = {2.0f, 9.0f, 4.0f, 1.0f, 7.0f};
    for (int i = 0; i < 5; i++) heap_push(h, i, dists[i]);

    for (int i = 0; i < 5; i++) {
        Candidate peeked, popped;
        assert(heap_peek(h, &peeked) == 1);
        int size_before = heap_size(h);

        assert(heap_pop(h, &popped) == 1);

        assert(peeked.id == popped.id);
        assert(peeked.dist == popped.dist);
        assert(heap_size(h) == size_before - 1);   // peek must not remove
    }

    heap_free(h);
    printf("  peek: PASS\n");
}

// --- Case 5: empty heap -----------------------------------------------
static void test_empty(void) {
    MaxHeap *h = heap_create(5);
    Candidate c;

    assert(heap_size(h) == 0);
    assert(heap_pop(h, &c) == 0);    // must fail, not crash
    assert(heap_peek(h, &c) == 0);

    // Push then drain then try again — the empty path after use.
    heap_push(h, 1, 1.0f);
    assert(heap_pop(h, &c) == 1);
    assert(heap_pop(h, &c) == 0);

    heap_free(h);
    printf("  empty: PASS\n");
}

// --- Case 6: full heap ------------------------------------------------
static void test_full(void) {
    int cap = 4;
    MaxHeap *h = heap_create(cap);

    for (int i = 0; i < cap; i++) assert(heap_push(h, i, (float)i) == 1);
    assert(heap_size(h) == cap);

    // The next push must be refused, and must not disturb the heap.
    assert(heap_push(h, 999, 0.5f) == 0);
    assert(heap_size(h) == cap);

    Candidate c;
    heap_peek(h, &c);
    assert(c.dist == 3.0f);          // still the original max
    assert(c.id != 999);             // the rejected element didn't sneak in

    heap_free(h);
    printf("  full: PASS\n");
}

// --- Case 7: single element -------------------------------------------
static void test_single(void) {
    MaxHeap *h = heap_create(5);
    Candidate c;

    heap_push(h, 42, 3.14f);
    assert(heap_size(h) == 1);
    assert(heap_peek(h, &c) == 1 && c.id == 42);
    assert(heap_pop(h, &c) == 1 && c.id == 42 && c.dist == 3.14f);
    assert(heap_size(h) == 0);

    heap_free(h);
    printf("  single: PASS\n");
}

// --- Case 8: deep tree, many elements ---------------------------------
// With few elements sift_down never recurses past one level. 100 random
// elements forces real multi-level sift paths in both directions.
static void test_deep(void) {
    int n = 100;
    MaxHeap *h = heap_create(n);

    srand(12345);                     // fixed seed: failures reproduce
    for (int i = 0; i < n; i++) {
        heap_push(h, i, (float)(rand() % 1000));
    }
    assert(heap_size(h) == n);

    Candidate prev, cur;
    heap_pop(h, &prev);
    for (int i = 1; i < n; i++) {
        assert(heap_pop(h, &cur) == 1);
        assert(cur.dist <= prev.dist);
        prev = cur;
    }

    heap_free(h);
    printf("  deep: PASS\n");
}

// --- Case 9: the actual top-k usage pattern ---------------------------
// This is how step 12 will use the heap: keep a bounded set of the k
// SMALLEST distances by evicting the largest whenever a better one shows up.
static void test_topk_pattern(void) {
    int k = 3;
    MaxHeap *h = heap_create(k);

    float stream[] = {50.0f, 10.0f, 80.0f, 30.0f, 5.0f, 90.0f, 20.0f};
    int n = 7;

    for (int i = 0; i < n; i++) {
        if (heap_size(h) < k) {
            heap_push(h, i, stream[i]);
        } else {
            Candidate worst;
            heap_peek(h, &worst);          // O(1) — the point of the heap
            if (stream[i] < worst.dist) {
                heap_pop(h, &worst);
                heap_push(h, i, stream[i]);
            }
        }
    }

    // The 3 smallest of the stream are 5, 10, 20.
    // Max-heap pops largest first.
    Candidate c;
    heap_pop(h, &c); assert(c.dist == 20.0f);
    heap_pop(h, &c); assert(c.dist == 10.0f);
    heap_pop(h, &c); assert(c.dist == 5.0f);
    assert(heap_size(h) == 0);

    heap_free(h);
    printf("  topk_pattern: PASS\n");
}

// --- Case 10: free(NULL) must be safe ---------------------------------
static void test_null_free(void) {
    heap_free(NULL);
    printf("  null_free: PASS\n");
}

int main(void) {
    printf("test_heap:\n");
    test_pop_order();
    test_insertion_orders();
    test_id_pairing();
    test_peek();
    test_empty();
    test_full();
    test_single();
    test_deep();
    test_topk_pattern();
    test_null_free();
    printf("test_heap: all passed\n");
    return 0;
}