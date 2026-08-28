#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "hnsw.h"

// --- Case 1: every level is in range ----------------------------------
static void test_range(void) {
    srand(42);
    double mL = 1.0 / log(16.0);

    for (int i = 0; i < 100000; i++) {
        int level = hnsw_random_level(mL);

        assert(level >= 0);
        assert(level <= HNSW_MAX_LEVEL);
    }

    printf("  range: PASS\n");
}

// --- Case 2: the distribution decays geometrically --------------------
// Each level should hold roughly 1/M of the level below. This is THE
// property that makes the hierarchy work — a constant ratio between
// layers is what gives logarithmic depth.
static void test_decay(void) {
    srand(42);
    int M = 16;
    double mL = 1.0 / log((double)M);
    int trials = 200000;

    int counts[HNSW_MAX_LEVEL + 1] = {0};
    for (int i = 0; i < trials; i++) {
        counts[hnsw_random_level(mL)]++;
    }

    printf("    M=%d, mL=%.4f, %d draws\n", M, mL, trials);
    for (int l = 0; l <= HNSW_MAX_LEVEL; l++) {
        if (counts[l] == 0) break;
        double pct = 100.0 * counts[l] / trials;
        printf("    level %2d: %7d  (%5.2f%%)", l, counts[l], pct);
        if (l > 0 && counts[l - 1] > 0) {
            printf("   ratio to level %d: 1/%.1f",
                   l - 1, (double)counts[l - 1] / counts[l]);
        }
        printf("\n");
    }

    float low = M/2;
    float high = M*2;
    assert(counts[0]/counts[1] >= low);
    assert(counts[0]/counts[1] <= high);

    assert(counts[1]/counts[2] >= low);
    assert(counts[1]/counts[2] <= high);

    printf("  decay: PASS\n");
}

// --- Case 3: the vast majority of nodes are level 0 -------------------
// If this weren't true, upper layers would be dense and the hierarchy
// would cost more than it saves.
static void test_mostly_zero(void) {
    srand(7);
    double mL = 1.0 / log(16.0);
    int trials = 100000;

    int zeros = 0;
    for (int i = 0; i < trials; i++) {
        if (hnsw_random_level(mL) == 0) zeros++;
    }

    double frac = (double)zeros / trials;
    printf("    level 0: %.2f%% of draws\n", frac * 100);

    assert(frac > 0.90);
    //       Expected is 1 - 1/M ≈ 93.75% for M=16.

    printf("  mostly_zero: PASS\n");
}

// --- Case 4: same seed, same sequence ---------------------------------
// Reproducible builds. Without this, a recall change can't be attributed
// to a code change versus a different random structure.
static void test_deterministic(void) {
    double mL = 1.0 / log(16.0);
    int n = 1000;
    int first[1000], second[1000];

    srand(123);
    for (int i = 0; i < n; i++) first[i] = hnsw_random_level(mL);

    srand(123);
    for (int i = 0; i < n; i++) second[i] = hnsw_random_level(mL);

    for (int i = 0; i < n; i++) assert(first[i] == second[i]);

    srand(3);
    int diff = 0;
    for (int i = 0; i < n; i++) {
        int random = hnsw_random_level(mL);
        if (first[i] != random) diff++;
    }

    assert(diff > 0);
    //printf("diff is %d\n", diff);

    printf("  deterministic: PASS\n");
}

// --- Case 5: larger mL produces higher levels -------------------------
// Confirms mL actually controls the decay rate rather than being ignored.
static void test_ml_controls_decay(void) {
    int trials = 50000;

    srand(1);
    double sum_small = 0;
    for (int i = 0; i < trials; i++) sum_small += hnsw_random_level(0.36);

    srand(1);
    double sum_large = 0;
    for (int i = 0; i < trials; i++) sum_large += hnsw_random_level(1.5);

    double mean_small = sum_small / trials;
    double mean_large = sum_large / trials;
    printf("    mean level: mL=0.36 -> %.3f    mL=1.5 -> %.3f\n",
           mean_small, mean_large);

    assert(mean_large > mean_small);
    printf("  ml_controls_decay: PASS\n");
}

// --- Case 6: the cap holds -------------------------------------------
// With a very large mL the raw draw would frequently exceed the cap.
static void test_cap(void) {
    srand(5);
    int hit_cap = 0;

    for (int i = 0; i < 20000; i++) {
        int level = hnsw_random_level(50.0);   // absurdly large mL
        assert(level <= HNSW_MAX_LEVEL);
        if (level == HNSW_MAX_LEVEL) hit_cap++;
    }

    printf("    draws hitting the cap with mL=50: %d of 20000\n", hit_cap);

    assert(hit_cap > 0);
    printf("  cap: PASS\n");
}

// --- Case 7: expected depth for a realistic dataset -------------------
// Not an assertion — a report. How many layers would 1M vectors produce?
// Useful sanity check that HNSW_MAX_LEVEL=16 is far above what's reachable.
static void test_report_expected_depth(void) {
    srand(99);
    double mL = 1.0 / log(16.0);
    int n = 1000000;

    int max_seen = 0;
    int counts[HNSW_MAX_LEVEL + 1] = {0};
    for (int i = 0; i < n; i++) {
        int l = hnsw_random_level(mL);
        counts[l]++;
        if (l > max_seen) max_seen = l;
    }

    printf("    simulating %d nodes, M=16:\n", n);
    for (int l = 0; l <= max_seen; l++) {
        printf("      layer %d would hold %d nodes\n", l, counts[l]);
    }
    printf("    max level reached: %d (cap is %d)\n", max_seen, HNSW_MAX_LEVEL);

    printf("  report_expected_depth: REPORTED\n");
}

int main(void) {
    printf("test_hnsw_levels:\n");
    test_range();
    test_decay();
    test_mostly_zero();
    test_deterministic();
    test_ml_controls_decay();
    test_cap();
    test_report_expected_depth();
    printf("test_hnsw_levels: all passed\n");
    return 0;
}