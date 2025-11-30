#define _POSIX_C_SOURCE 199309L
#include "bfs.h"

#include "mat_to_csr.h"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/cilkscale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SAMPLING_ROUNDS 2
#define SAMPLING_K 2

/* Return current time in seconds with sub-second precision. */
static double now_seconds(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + t.tv_nsec * 1e-9;
}

//  ------ IMPLEMENT REDUCER FOR VARIABLE:'CHANGED' -------
void bool_or_identity(void *view) { *(int *)view = 0; }

void bool_or_reduce(void *left, void *right) {
  int a = *(int *)left;
  int b = *(int *)right;
  *(int *)left = a | b;
}
//  --------------------------------------------

int coloring_cc(int n, const int *row_ptr, const int *col_idx, int *labels,
                int max_iters) {
  if (!row_ptr || !col_idx || !labels)
    return -1;

  // Inform compiler about alignment for better vectorization
  row_ptr = __builtin_assume_aligned(row_ptr, 64);
  col_idx = __builtin_assume_aligned(col_idx, 64);
  labels = __builtin_assume_aligned(labels, 64);

  if (n <= 0)
    return 0;
  /* initialize each vertex label to its own id (1-based) per assignment
   * requirement: labels range from 1..n instead of 0..n-1 */
  for (int v = 0; v < n; ++v)
    labels[v] = v + 1;

  int iterations = 0;
  int cilk_reducer(bool_or_identity, bool_or_reduce) changed = 1;

  // Configuration from Environment
  static int sampling_rounds = 0;
  static int sampling_k = 2;
  static int prefetch_dist = 16;
  static int zero_planting = 1;
  static int config_loaded = 0;

  if (!config_loaded) {
    char *env_rounds = getenv("BFS_SAMPLING_ROUNDS");
    if (env_rounds)
      sampling_rounds = atoi(env_rounds);

    char *env_k = getenv("BFS_SAMPLING_K");
    if (env_k)
      sampling_k = atoi(env_k);

    char *env_dist = getenv("BFS_PREFETCH_DIST");
    if (env_dist)
      prefetch_dist = atoi(env_dist);

    char *env_zp = getenv("BFS_ZERO_PLANTING");
    if (env_zp)
      zero_planting = atoi(env_zp);

    printf("BFS Config: Sampling Rounds=%d, K=%d, Prefetch Dist=%d, Zero "
           "Planting=%d\n",
           sampling_rounds, sampling_k, prefetch_dist, zero_planting);
    config_loaded = 1;
  }

  // Optimization: Zero Planting
  // Find the node with the highest degree and give it the smallest label (1).
  // This helps propagate the 'winning' label faster.
  if (zero_planting && iterations == 0) {
    int max_v = 0;
    int max_deg = -1;

    // Sequential scan is fast enough (memory bandwidth bound)
    for (int i = 0; i < n; i++) {
      int deg = row_ptr[i + 1] - row_ptr[i];
      if (deg > max_deg) {
        max_deg = deg;
        max_v = i;
      }
    }

    if (max_v != 0) {
      // Swap labels[0] (which is 1) and labels[max_v]
      int temp = labels[0];
      labels[0] = labels[max_v];
      labels[max_v] = temp;

      // Initial Push: Propagate label 1 to all neighbors of max_v
      int start = row_ptr[max_v];
      int end = row_ptr[max_v + 1];

      // Calculate average degree for Hub Push
      int avg_deg = (n > 0) ? (row_ptr[n] / n) : 0;

      cilk_for(int e = start; e < end; e++) {
        int u = col_idx[e];
        if (u >= 0 && u < n) {
          // We know labels[max_v] is now 1 (or whatever labels[0] was, usually
          // 1) But strictly speaking we swapped. labels[max_v] has the small
          // label. We propagate labels[max_v] to neighbors.
          int val = labels[max_v];
          if (val < labels[u]) {
            __atomic_fetch_min(&labels[u], val, __ATOMIC_SEQ_CST);

            // Hub Push: If neighbor u is a hub, push to its neighbors too
            int u_deg = row_ptr[u + 1] - row_ptr[u];
            if (u_deg > avg_deg) {
              int u_start = row_ptr[u];
              int u_end = row_ptr[u + 1];
              for (int k = u_start; k < u_end; k++) {
                int w = col_idx[k];
                if (w >= 0 && w < n) {
                  if (val < labels[w]) {
                    __atomic_fetch_min(&labels[w], val, __ATOMIC_SEQ_CST);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  wsp_t start, end;
  while (changed) {
    ++iterations;
    changed = 0;
    /* For each vertex, try to adopt a smaller label seen among neighbors */
    start = wsp_getworkspan();
    cilk_for(int v = 0; v < n; v++) {
      int best = labels[v];

      // Optimization: Early Retirement / Zero Convergence
      // If we already have the global minimum label (1), we are done.
      if (best == 1)
        continue;

      // Optimization: Path compression / Shortcut (grandparent traversal)
      while (best > labels[best - 1]) {
        best = labels[best - 1];
      }

      // Optimization: Global Pointer Jumping / Early Update
      if (best < labels[v]) {
        int original_label =
            __atomic_fetch_min(&labels[v], best, __ATOMIC_SEQ_CST);
        if (original_label > best)
          changed = 1;
      }

      // Optimization: Early exit if we found the global minimum
      if (best == 1)
        continue;

      int start_edge = row_ptr[v];
      int end_edge = row_ptr[v + 1];
      int limit = end_edge;

      // Optimization: Neighbor Sampling
      if (iterations <= sampling_rounds) {
        if (start_edge + sampling_k < end_edge)
          limit = start_edge + sampling_k;
      }

      for (int e = start_edge; e < limit; ++e) {
        // Optimization: Software Prefetching
        if (prefetch_dist > 0) {
          // Stage 1: Prefetch the neighbor index (col_idx) further ahead
          if (e + prefetch_dist * 2 < limit) {
            __builtin_prefetch(&col_idx[e + prefetch_dist * 2], 0, 3);
          }
          // Stage 2: Prefetch the label using the (hopefully) now-cached
          // neighbor index
          if (e + prefetch_dist < limit) {
            int u_next = col_idx[e + prefetch_dist];
            if (u_next >= 0 && u_next < n) {
              __builtin_prefetch(&labels[u_next], 0, 3);
            }
          }
        }

        int u = col_idx[e];
        /* skip invalid neighbor indices */
        if (u < 0 || u >= n)
          continue;

        int lu = labels[u];
        if (lu < best) {
          best = lu;
          if (best == 1)
            break; /* Optimization: Early exit */
        }
      }

      /* Optimization: Check before atomic to avoid cache line invalidation */
      if (best < labels[v]) {
        int original_label =
            __atomic_fetch_min(&labels[v], best, __ATOMIC_SEQ_CST);
        if (original_label > best) {
          changed = 1;
        }
      }
    }
    end = wsp_getworkspan();
    if (max_iters > 0 && iterations >= max_iters)
      break;
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
  if (n <= 0 || !labels)
    return 0;

  /* labels are expected in the range 1..n (inclusive) */
  unsigned char *seen = calloc((size_t)(n + 1), 1); /* index 0 unused */
  if (!seen)
    return -1;

  int count = 0;
  for (int i = 0; i < n; ++i) {
    int lbl = labels[i];
    if (lbl < 1 || lbl > n)
      continue; /* skip invalid labels */
    if (!seen[lbl]) {
      seen[lbl] = 1;
      ++count;
    }
  }

  free(seen);
  return count;
}

/* Print up to `max_print` label pairs (vertex:label) for quick inspection */
void print_sample_labels(int n, const int *labels, int max_print) {
  int limit = (n < max_print) ? n : max_print;
  /* Print vertex indices 1-based to match label numbering */
  for (int i = 0; i < limit; ++i)
    printf("%d:%d ", i + 1, labels[i]);
  if (limit > 0)
    printf("\n");
}

/*
 * High-level wrapper: allocate labels, run coloring_cc, measure time,
 * and return allocated labels via out_labels. Caller must free *out_labels.
 * Returns 0 on success, negative on error.
 */
int run_coloring_cc_from_csr(int n, const int *row_ptr, const int *col_idx,
                             int **out_labels, int max_iters,
                             double *elapsed_seconds, int *out_iters) {

  if (!out_labels)
    return -1;

  if (n <= 0 || !row_ptr || !col_idx)
    return -2;

  int *labels = (int *)malloc((size_t)n * sizeof(int));

  if (!labels)
    return -3;

  double t0 = now_seconds();
  int iters = coloring_cc(n, row_ptr, col_idx, labels, max_iters);
  double t1 = now_seconds();

  *out_labels = labels;
  if (elapsed_seconds)
    *elapsed_seconds = t1 - t0;

  if (out_iters)
    *out_iters = iters;

  return 0;
}
