#ifndef BFS_H
#define BFS_H

// ============================================================================
// BFS/Connected Components Algorithms (OpenCilk Implementation)
// ============================================================================

// Configuration Constants
#define BFS_SAMPLING_ROUNDS_DEFAULT 2
#define BFS_SAMPLING_K_DEFAULT 2

// ============================================================================
// Algorithm API
// ============================================================================

/**
 * Adaptive Connected Components dispatcher.
 * Selects optimal algorithm (BFS or Label Propagation) based on graph density.
 *
 * @param n         Number of vertices
 * @param row_ptr   CSR row pointers (size: n+1)
 * @param col_idx   CSR column indices
 * @param labels    Output labels array (size: n), will be filled with component
 * IDs
 * @param max_iters Maximum iterations (0 = unlimited)
 * @return          Number of iterations performed, or negative on error
 */
int coloring_cc(int n, const long long *row_ptr, const int *col_idx,
                int *labels, int max_iters);

/**
 * Frontier-based parallel BFS for Connected Components.
 * Best for low-density / large-diameter graphs (e.g., k-mer graphs).
 */
int coloring_cc_bfs(int n, const long long *row_ptr, const int *col_idx,
                    int *labels, int max_iters);

/**
 * Label Propagation for Connected Components.
 * Best for high-density / small-diameter graphs (e.g., social networks).
 */
int coloring_cc_lp(int n, const long long *row_ptr, const int *col_idx,
                   int *labels, int max_iters);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Count the number of distinct connected components.
 */
int count_components(int n, const int *labels);

/**
 * Print sample labels for debugging.
 */
void print_sample_labels(int n, const int *labels, int max_print);

/**
 * High-level wrapper: allocate labels, run coloring_cc, measure time.
 * Caller must free *out_labels.
 *
 * @return 0 on success, negative on error
 */
int run_coloring_cc_from_csr(int n, const long long *row_ptr,
                             const int *col_idx, int **out_labels,
                             int max_iters, double *elapsed_seconds,
                             int *out_iters);

#endif /* BFS_H */
