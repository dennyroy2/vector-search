#include "visited.h"
#include <stdlib.h>

VisitedSet *visited_create(int n) {
    VisitedSet *v = malloc(sizeof(*v));
    if (!v) return NULL;

    v->stamp = calloc(n, sizeof(*v->stamp));
    if (!v->stamp) { free(v); return NULL; }
    v->generation = 0;
    v->n = n;
    return v;
}

void visited_free(VisitedSet *v) {
    if (!v) return;
    free(v->stamp);
    free(v);
}


void visited_reset(VisitedSet *v) {
    v->generation++;
}

int visited_check(const VisitedSet *v, int node) {
    return v->stamp[node] == v->generation;
}

void visited_mark(VisitedSet *v, int node) {
    v->stamp[node] = v->generation;
}