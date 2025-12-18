/**
 * @file BFScilk.c
 * @brief Shared-memory Connected Components using OpenCilk
 *
 * This file implements two parallel algorithms for finding connected
 * components:
 *   1. Frontier-Based BFS - Best for sparse, large-diameter graphs
 *   2. Label Propagation - Best for dense, small-diameter graphs
 *
 * An adaptive dispatcher (coloring_cc) selects the algorithm based on average
 * degree.
 *
 * See ARCHITECTURE.md for detailed algorithm descriptions and optimizations.
 *
 * Key optimizations:
 *   - Zero Planting: Start from highest-degree vertex
 *   - Hub Push: Propagate through high-degree neighbors
 *   - Path Compression: Follow label chains
 *   - Sampling: Limit neighbor exploration early on
 *   - Software Prefetching: Hide memory latency
 */
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

/* Default configuration (overridden by environment variables) */
#define SAMPLING_ROUNDS 2
#define SAMPLING_K 2

/* ========================================================================
 * Utility Functions
 * ======================================================================== */

/** High-precision timer for benchmarking. */
static double now_seconds(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + t.tv_nsec * 1e-9;
}

/* ========================================================================
 * OpenCilk Reducer for 'changed' flag
 * ========================================================================
 * A reducer allows race-free accumulation across parallel iterations.
 * This reducer implements a boolean OR: any thread setting changed=1
 * will cause the final result to be 1.
 */
void bool_or_identity(void *view) { *(int *)view = 0; }

void bool_or_reduce(void *left, void *right) {
  int a = *(int *)left;
  int b = *(int *)right;
  *(int *)left = a | b;
}

/* ========================================================================
 * Frontier-Based BFS Algorithm
 * ========================================================================
 * Classic BFS starting from each unvisited vertex.
 * - O(V + E) work total across all components
 * - O(diameter) iterations per component
 * - Parallelism over frontier vertices via cilk_for
 *
 * Best for: Sparse graphs, graphs with many components
 */
int coloring_cc_bfs(int n, const long long *row_ptr, const int *col_idx,
                    int *labels, int max_iters) {
  if (!row_ptr || !col_idx || !labels)
    return -1;

  /* Inform compiler about alignment for SIMD vectorization */
  row_ptr = __builtin_assume_aligned(row_ptr, 64);
  col_idx = __builtin_assume_aligned(col_idx, 64);
  labels = __builtin_assume_aligned(labels, 64);

  if (n <= 0)
    return 0;

  /* Initialize: 0 = unvisited, >0 = visited (component ID) */
  memset(labels, 0, n * sizeof(int));

  /* Double-buffered frontiers for BFS levels */
  int *frontier = (int *)malloc(n * sizeof(int));
  int *next_frontier = (int *)malloc(n * sizeof(int));

  if (!frontier || !next_frontier) {
    free(frontier);
    free(next_frontier);
    return -1;
  }

  int total_levels = 0;

  /* Outer loop: Find each connected component */
  for (int i = 0; i < n; ++i) {
    if (labels[i] == 0) {
      /* Start new component from unvisited vertex i */
      int component_id = i + 1; /* 1-based component IDs */
      labels[i] = component_id;

      frontier[0] = i;
      int frontier_size = 1;

      /* BFS expansion for this component */
      while (frontier_size > 0) {
        total_levels++;
        int next_frontier_size = 0;

        /* Parallel frontier expansion */
        cilk_for(int f = 0; f < frontier_size; ++f) {
          int u = frontier[f];
          long long start = row_ptr[u];
          long long end = row_ptr[u + 1];

          for (long long e = start; e < end; ++e) {
            int v = col_idx[e];

            /* Atomically claim unvisited neighbor */
            if (labels[v] == 0) {
              if (__atomic_compare_exchange_n(&labels[v], &(int){0},
                                              component_id, 0, __ATOMIC_RELAXED,
                                              __ATOMIC_RELAXED)) {
                /* Reserve slot in next frontier */
                int idx = __atomic_fetch_add(&next_frontier_size, 1,
                                             __ATOMIC_RELAXED);
                next_frontier[idx] = v;
              }
            }
          }
        }

        /* Swap frontier buffers */
        int *temp = frontier;
        frontier = next_frontier;
        next_frontier = temp;
        frontier_size = next_frontier_size;
      }
    }
  }

  free(frontier);
  free(next_frontier);

  printf("OpenCilk BFS (Frontier-Based) using %d workers(cores)\n",
         __cilkrts_get_nworkers());

  return total_levels;
}

/* ========================================================================
 * Label Propagation Algorithm
 * ========================================================================
 * Iteratively propagates minimum labels until convergence.
 * - O(D × E) work where D = diameter
 * - O(D) iterations
 * - Full parallelism over all vertices via cilk_for
 *
 * Best for: Dense graphs with small diameter (social networks)
 */
int coloring_cc_lp(int n, const long long *row_ptr, const int *col_idx,
                   int *labels, int max_iters) {
  if (!row_ptr || !col_idx || !labels)
    return -1;

  row_ptr = __builtin_assume_aligned(row_ptr, 64);
  col_idx = __builtin_assume_aligned(col_idx, 64);
  labels = __builtin_assume_aligned(labels, 64);

  if (n <= 0)
    return 0;

  /* Initialize: each vertex is its own component (1-based) */
  for (int v = 0; v < n; ++v)
    labels[v] = v + 1;

  int iterations = 0;
  int cilk_reducer(bool_or_identity, bool_or_reduce) changed = 1;

  /* Load configuration from environment (once) */
  static int sampling_rounds = 0;
  static int sampling_k = 2;
  static int prefetch_dist = 16;
  static int zero_planting = 1;
  static int hub_push = 1;
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

    char *env_hub_push = getenv("BFS_HUB_PUSH");
    if (env_hub_push)
      hub_push = atoi(env_hub_push);

    printf("OpenCilk BFS Config: Sampling Rounds=%d, K=%d, Prefetch Dist=%d, "
           "Zero Planting=%d, Hub Push=%d\n",
           sampling_rounds, sampling_k, prefetch_dist, zero_planting, hub_push);
    config_loaded = 1;
  }

  /* ====== Zero Planting Optimization ======
   * Find the highest-degree vertex and swap labels so it gets label 1.
   * This accelerates convergence since label 1 will propagate from the
   * most connected vertex.
   */
  if (zero_planting && iterations == 0) {
    int max_v = 0;
    int max_deg = -1;

    for (int i = 0; i < n; i++) {
      long long deg = row_ptr[i + 1] - row_ptr[i];
      if (deg > max_deg) {
        max_deg = deg;
        max_v = i;
      }
    }

    if (max_v != 0) {
      /* Swap labels[0] and labels[max_v] */
      int temp = labels[0];
      labels[0] = labels[max_v];
      labels[max_v] = temp;

      long long start = row_ptr[max_v];
      long long end = row_ptr[max_v + 1];
      int avg_deg = (n > 0) ? (row_ptr[n] / n) : 0;

      /* Push label 1 to all neighbors of the max-degree vertex */
      cilk_for(long long e = start; e < end; e++) {
        int u = col_idx[e];
        if (u >= 0 && u < n) {
          int val = labels[max_v];
          if (val < labels[u]) {
            __atomic_fetch_min(&labels[u], val, __ATOMIC_SEQ_CST);

            /* Hub Push: if neighbor is also a hub, push to its neighbors */
            if (hub_push) {
              long long u_deg = row_ptr[u + 1] - row_ptr[u];
              if (u_deg > avg_deg) {
                long long u_start = row_ptr[u];
                long long u_end = row_ptr[u + 1];
                for (long long k = u_start; k < u_end; k++) {
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
  }

  /* ====== Main Propagation Loop ====== */
  wsp_t start, end;
  start = wsp_getworkspan();
  while (changed) {
    ++iterations;
    changed = 0;

    cilk_for(int v = 0; v < n; v++) {
      int best = labels[v];

      /* Already at global minimum, skip */
      if (best == 1)
        continue;

      /* Path Compression: follow label chain to find true minimum */
      while (best > labels[best - 1]) {
        best = labels[best - 1];
      }

      /* Update own label if compressed */
      if (best < labels[v]) {
        int original_label =
            __atomic_fetch_min(&labels[v], best, __ATOMIC_SEQ_CST);
        if (original_label > best)
          changed = 1;
      }

      if (best == 1)
        continue;

      /* Sampling: limit neighbor exploration in early rounds */
      long long start_edge = row_ptr[v];
      long long end_edge = row_ptr[v + 1];
      long long limit = end_edge;

      if (iterations <= sampling_rounds) {
        if (start_edge + sampling_k < end_edge)
          limit = start_edge + sampling_k;
      }

      /* Scan neighbors for smaller labels */
      for (long long e = start_edge; e < limit; ++e) {
        /* Two-stage software prefetching */
        if (prefetch_dist > 0) {
          if (e + prefetch_dist * 2 < limit) {
            __builtin_prefetch(&col_idx[e + prefetch_dist * 2], 0, 3);
          }
          if (e + prefetch_dist < limit) {
            int u_next = col_idx[e + prefetch_dist];
            if (u_next >= 0 && u_next < n) {
              __builtin_prefetch(&labels[u_next], 0, 3);
            }
          }
        }

        int u = col_idx[e];
        if (u < 0 || u >= n)
          continue;

        int lu = labels[u];
        if (lu < best) {
          best = lu;
          if (best == 1)
            break; /* Can't get better than global minimum */
        }
      }

      /* Update label if we found something better */
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

  printf("OpenCilk BFS (Label Propagation) using %d workers(cores)\n",
         __cilkrts_get_nworkers());
  wsp_t elapsed = wsp_sub(end, start);
  wsp_dump(elapsed, "BFS-OpenCilk-LP");

  return iterations;
}

/* ========================================================================
 * Adaptive Algorithm Dispatcher
 * ======================================================================== */
int coloring_cc(int n, const long long *row_ptr, const int *col_idx,
                int *labels, int max_iters) {
  if (n <= 0)
    return 0;

  /* Heuristic: Use average degree to select algorithm.
   * High density (AvgDeg > 4) → small diameter → Label Propagation
   * Low density (AvgDeg ≤ 4) → large diameter → Frontier BFS
   * Threshold 4.0 is empirical (LiveJournal ~14, Kmer ~2)
   */
  long long nz = row_ptr[n];
  double avg_deg = (double)nz / n;

  if (avg_deg > 4.0) {
    printf(
        "Adaptive BFS: High density (AvgDeg=%.2f) -> Using Label Propagation\n",
        avg_deg);
    return coloring_cc_lp(n, row_ptr, col_idx, labels, max_iters);
  } else {
    printf("Adaptive BFS: Low density (AvgDeg=%.2f) -> Using Frontier BFS\n",
           avg_deg);
    return coloring_cc_bfs(n, row_ptr, col_idx, labels, max_iters);
  }
}

/* ========================================================================
 * Utility Functions
 * ======================================================================== */

/**
 * Count distinct connected components.
 * Uses a simple bitmap approach (O(n) space).
 */
int count_components(int n, const int *labels) {
  if (n <= 0 || !labels)
    return 0;

  unsigned char *seen = calloc((size_t)(n + 1), 1);
  if (!seen)
    return -1;

  int count = 0;
  for (int i = 0; i < n; ++i) {
    int lbl = labels[i];
    if (lbl < 1 || lbl > n)
      continue;
    if (!seen[lbl]) {
      seen[lbl] = 1;
      ++count;
    }
  }

  free(seen);
  return count;
}

/** Print sample labels for debugging. */
void print_sample_labels(int n, const int *labels, int max_print) {
  int limit = (n < max_print) ? n : max_print;
  for (int i = 0; i < limit; ++i)
    printf("%d:%d ", i + 1, labels[i]);
  if (limit > 0)
    printf("\n");
}

/**
 * High-level wrapper: runs coloring_cc and measures time.
 * Allocates labels array - caller must free *out_labels.
 */
int run_coloring_cc_from_csr(int n, const long long *row_ptr,
                             const int *col_idx, int **out_labels,
                             int max_iters, double *elapsed_seconds,
                             int *out_iters) {

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
