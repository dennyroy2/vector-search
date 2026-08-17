#include <assert.h>
#include <stdio.h>
#include "distance.h"
#include <math.h>

static int close(float got, float expected) {
    return fabsf(got - expected) < 1e-4f;
}

int main(void) {

    {
        float a[] = {0.0f, 0.0f};
        float b[] = {3.0f, 4.0f};
        float d = l2sq_distance(a, b, 2);
        assert(close(d, 25.0f));
    }

    {
        float a[] = {1.5f, -2.25f, 88.0f, 0.0f};
        assert(l2sq_distance(a, a, 4) == 0.0f);
    }


    // --- Case 3: symmetry ------------------------------------------------
    // Squaring discards the sign, so d(a,b) and d(b,a) are bit-identical.
    {
        float a[] = {1.0f, 2.0f, 3.0f};
        float b[] = {4.0f, 6.0f, 8.0f};
        assert(l2sq_distance(a, b, 3) == l2sq_distance(b, a, 3));
    }

    // --- Case 4: dim = 1, the degenerate case ----------------------------
    // (5-2)^2 = 9
    {
        float a[] = {5.0f};
        float b[] = {2.0f};
        assert(close(l2sq_distance(a, b, 1), 9.0f));
    }

    // --- Case 5: negatives, to confirm squaring handles sign -------------
    // (-1 - 2)^2 + (-4 - -1)^2 = 9 + 9 = 18
    {
        float a[] = {-1.0f, -4.0f};
        float b[] = { 2.0f, -1.0f};
        assert(close(l2sq_distance(a, b, 2), 18.0f));
    }

    // --- Case 6: non-integer values, hand-computed -----------------------
    // (0.5-1.5)^2 + (2.25-0.25)^2 + (3.0-3.5)^2
    //   = 1.0 + 4.0 + 0.25 = 5.25
    {
        float a[] = {0.5f, 2.25f, 3.0f};
        float b[] = {1.5f, 0.25f, 3.5f};
        assert(close(l2sq_distance(a, b, 3), 5.25f));
    }

    // --- Case 7: ordering is what actually matters -----------------------
    // The whole point of this function is comparing distances, so test
    // that a nearer vector really does score lower.
    {
        float q[]    = {0.0f, 0.0f};
        float near[] = {1.0f, 0.0f};
        float far[]  = {5.0f, 5.0f};
        assert(l2sq_distance(q, near, 2) < l2sq_distance(q, far, 2));
    }
    
    return 0;
}