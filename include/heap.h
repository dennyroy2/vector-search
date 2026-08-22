#ifndef HEAP_H
#define HEAP_H
typedef struct {
    int id;
    float dist;
} Candidate;

typedef struct {
    Candidate * items;
    int size;
    int capacity;
    int is_max;
} MaxHeap;

MaxHeap *heap_create(int capacity, int is_max);
void heap_free(MaxHeap *h);

// Returns 1 on success, 0 if full.
int heap_push(MaxHeap *h, int id, float dist);

// Removes and returns the farthest element. 1 on success, 0 if empty.
int heap_pop(MaxHeap *h, Candidate *out);

// Reads the farthest element without removing it. O(1).
int heap_peek(const MaxHeap *h, Candidate *out);

int heap_size(const MaxHeap *h);

#endif