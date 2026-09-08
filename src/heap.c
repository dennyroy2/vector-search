#include <stdlib.h>
#include "heap.h"

MaxHeap * heap_create(int capacity, int is_max) {

    MaxHeap * heap = malloc(sizeof(MaxHeap));
    if (heap == NULL) return NULL;

    heap->items = malloc(capacity * sizeof(Candidate));

    if (heap->items == NULL) {
        free(heap);
        return NULL;
    }
    heap->size = 0;
    heap->capacity = capacity;
    heap->is_max = is_max;

    return heap;
}

void heap_free(MaxHeap *h) {
    if (!h) return;
    free(h->items);
    free(h);
}

static void swap(Candidate * a, Candidate * b) {
    Candidate temp = *a;
    *a = *b;
    *b = temp;
}

static void sift_up(MaxHeap * h, int i) {
    int max = h->is_max;
    while(i > 0) {
        int parent = (i-1)/2;
        if (max) {
            if (h->items[i].dist <= h->items[parent].dist) break;
        } else {
            if (h->items[i].dist >= h->items[parent].dist) break;
        }

        swap(&h->items[i], &h->items[parent]);
        i = parent;
    }
}

static void sift_down(MaxHeap *h, int i) {
    int max = h->is_max;
    while (1) {
        int left  = 2*i + 1;
        int right = 2*i + 2;
        int best  = i;

        if (max) {
            if (left  < h->size && h->items[left].dist  > h->items[best].dist) best = left;
            if (right < h->size && h->items[right].dist > h->items[best].dist) best = right;
        } else {
            if (left  < h->size && h->items[left].dist  < h->items[best].dist) best = left;
            if (right < h->size && h->items[right].dist < h->items[best].dist) best = right;
        }

        if (best == i) break;             // already in the right place

        swap(&h->items[i], &h->items[best]);
        i = best;
    }
}

int heap_push(MaxHeap *h, int id, float dist) {
    if (h->size >= h->capacity) return 0;      // caller must handle "full"

    h->items[h->size].dist = dist;    // place at the end
    h->items[h->size].id = id;
    h->size++;
    sift_up(h, h->size - 1);      // let it rise
    return 1;
}

int heap_pop(MaxHeap *h, Candidate * out) {
    if (h->size == 0) return 0;

    *out = h->items[0];
    h->items[0] = h->items[h->size - 1];
    h->size--;
    sift_down(h, 0);
    return 1;
}

int heap_peek(const MaxHeap * h, Candidate * out) {
    if (h->size == 0) return 0;
    *out = h->items[0];
    return 1;
}

int heap_size(const MaxHeap *h) {
    return h->size;
}

void heap_reset(MaxHeap * h) {
    h->size = 0;
}