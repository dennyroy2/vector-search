#ifndef VISITED_H
#define VISITED_H

typedef struct {
    int * stamp;
    int generation;
    int n;
} VisitedSet;

VisitedSet *visited_create(int n);
void visited_free(VisitedSet *v);
void visited_reset(VisitedSet *v);          // begin a new search — O(1)
int  visited_check(const VisitedSet *v, int node);
void visited_mark(VisitedSet *v, int node);
#endif