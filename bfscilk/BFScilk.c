#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mat_to_csr.h"
#include <cilk/cilk.h>
#include <cilk/cilkscale.h>
#include <cilk/cilk_api.h>
#include "bfs.h"

/* Return current time in seconds with sub-second precision. */
static double now_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + t.tv_nsec * 1e-9;
}

//  ------ IMPLEMENT REDUCER FOR VARIABLE:'CHANGED' -------
void bool_or_identity(void* view){
    *(int* )view=0;
}

void bool_or_reduce(void* left, void* right){
    int a = *(int* )left;  
    int b = *(int* )right;
    *(int* )left = a | b;
}
//  --------------------------------------------


int coloring_cc(int n, const int *row_ptr, const int *col_idx, int *labels, int max_iters) {
    if (!row_ptr || !col_idx || !labels) return -1;
    
    // Inform compiler about alignment for better vectorization
    row_ptr = __builtin_assume_aligned(row_ptr, 64);
    col_idx = __builtin_assume_aligned(col_idx, 64);
    labels = __builtin_assume_aligned(labels, 64);

    if (n <= 0) return 0;
    /* initialize each vertex label to its own id (1-based) per assignment
     * requirement: labels range from 1..n instead of 0..n-1 */
    for (int v = 0; v < n; ++v) labels[v] = v + 1;

    int iterations = 0;
    int cilk_reducer(bool_or_identity, bool_or_reduce) changed = 1;
            
    wsp_t start,end;
    while (changed) {
            ++iterations;
            changed = 0;
            /* For each vertex, try to adopt a smaller label seen among neighbors */
            start = wsp_getworkspan();
            cilk_for (int v = 0; v < n; v++) {
                int best = labels[v];
                
                /* Optimization: Path compression / Shortcut (grandparent traversal)
                 * If the component leader I point to has found a smaller leader, update myself.
                 * This follows the label chain: best -> labels[best-1] -> labels[labels[best-1]-1]
                 * which effectively uses the 'grandparent' label (i.e., the label of my current label)
                 * to shortcut and flatten the label tree, reducing the number of iterations.
                 */
                while (best > labels[best-1]) {
                    best = labels[best-1];
                }

                /* Optimization: Early exit if we found the global minimum */
                if (best == 1) {
                    if (best < labels[v]) {
                        int original_label = __atomic_fetch_min(&labels[v], best, __ATOMIC_SEQ_CST);
                        if (original_label > best) changed = 1;
                    }
                    continue;
                }

                for (int e = row_ptr[v]; e < row_ptr[v+1]; ++e) {
                    int u = col_idx[e];
                    /* skip invalid neighbor indices (defensive, in case .mat wasn't strictly adjacency) */
                    if (u < 0 || u >= n) continue;
                    int lu = labels[u];
                    if (lu < best) {
                        best = lu;
                        if (best == 1) break; /* Optimization: Early exit */
                    }
                }
                
                /* Optimization: Check before atomic to avoid cache line invalidation */
                if (best < labels[v]) {
                    int original_label = __atomic_fetch_min(&labels[v], best, __ATOMIC_SEQ_CST);
                    if (original_label > best) {      
                        changed=1;
                    }   
                 }
        }
                end = wsp_getworkspan();
        if (max_iters > 0 && iterations >= max_iters) break;
    }
        printf("OpenCilk using %d workers(cores)\n", __cilkrts_get_nworkers());
    wsp_t elapsed = wsp_sub(end, start);
        wsp_dump(elapsed, "BFS-OpenCilk");

    return iterations;
}



/*
 * count_components
 *
 * A naive method that counts how many distinct labels appear in `labels`.
 * This assumes labels are in the range [0, n-1]. For very large graphs this
 * could use a hash set or sort+unique, but calloc'ing an n-sized bitmap is
 * simplest and fine for moderate-sized inputs.
 */
int count_components(int n, const int *labels) {
    if (n <= 0 || !labels) return 0;

    /* labels are expected in the range 1..n (inclusive) */
    unsigned char *seen = calloc((size_t)(n + 1), 1); /* index 0 unused */
    if (!seen) return -1;

    int count = 0;
    for (int i = 0; i < n; ++i) {
        int lbl = labels[i];
        if (lbl < 1 || lbl > n) continue; /* skip invalid labels */
        if (!seen[lbl]) { seen[lbl] = 1; ++count; }
    }

    free(seen);
    return count;
}

/* Print up to `max_print` label pairs (vertex:label) for quick inspection */
void print_sample_labels(int n, const int *labels, int max_print) {
    int limit = (n < max_print) ? n : max_print;
    /* Print vertex indices 1-based to match label numbering */
    for (int i = 0; i < limit; ++i) printf("%d:%d ", i + 1, labels[i]);
    if (limit > 0) printf("\n");
}

/*
 * High-level wrapper: allocate labels, run coloring_cc, measure time,
 * and return allocated labels via out_labels. Caller must free *out_labels.
 * Returns 0 on success, negative on error.
 */
int run_coloring_cc_from_csr(int n, const int *row_ptr, const int *col_idx,
    int **out_labels, int max_iters,double *elapsed_seconds, int *out_iters){
    
    if (!out_labels) return -1;
    
    if (n <= 0 || !row_ptr || !col_idx) return -2;

    int *labels = (int*) malloc((size_t)n * sizeof(int));
    
    if (!labels) return -3;

    double t0 = now_seconds();
    int iters = coloring_cc(n, row_ptr, col_idx, labels, max_iters);
    double t1 = now_seconds();

    *out_labels = labels;
    if (elapsed_seconds) *elapsed_seconds = t1 - t0;
    
    if (out_iters) *out_iters = iters;
    
    return 0;
}
