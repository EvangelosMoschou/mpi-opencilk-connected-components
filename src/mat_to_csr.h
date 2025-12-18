#ifndef MAT_TO_CSR_H
#define MAT_TO_CSR_H

#include <stddef.h>
#include <stdint.h>

// ============================================================================
// CSR (Compressed Sparse Row) Graph Representation
// ============================================================================

typedef struct {
  long long nzcount; // Number of non-zero entries (edges)
  int nrows;         // Number of rows (vertices)

  long long *row_ptr; // Row pointers (size: nrows + 1)
  int *col_idx;       // Column indices (size: nzcount)
} CSR_info;

// ============================================================================
// Node Degree Struct (for degree-based reordering)
// ============================================================================

typedef struct {
  int id;     // Original vertex ID
  int degree; // Vertex degree
} NodeDegree;

// ============================================================================
// Memory Utilities
// ============================================================================

/**
 * Aligned allocation wrapper (64-byte alignment for AVX-512/Cache lines).
 * Returns NULL on failure.
 */
void *aligned_alloc_wrapper(size_t size);

/**
 * Free CSR structure memory.
 */
void clean_csr_vectors(CSR_info vecs);

// ============================================================================
// MAT File Reading API
// ============================================================================

/**
 * Read a MATLAB .mat file and convert to CSR format.
 * Automatically handles:
 *   - v7.3 HDF5 files (chunked streaming for large graphs)
 *   - Symmetrization (A = A ∪ A^T)
 *   - Degree-based vertex reordering
 *   - Binary caching (.csrbin)
 *
 * @param file_name  Path to .mat file
 * @return           CSR_info structure (check nrows > 0 for success)
 */
CSR_info mat_to_csr(char *file_name);

// ============================================================================
// Distributed Loading API (for MPI)
// ============================================================================

/**
 * Compute edge-balanced partition for distributed loading.
 *
 * @param fname   Path to .csrbin file
 * @param size    Number of MPI processes
 * @param out_n   Output: total number of vertices
 * @return        Array of size (size+1) with partition boundaries, or NULL on
 * error
 */
int *compute_balanced_partition(const char *fname, int size, int *out_n);

/**
 * Read a partial CSR binary file for distributed loading.
 *
 * @param fname       Path to .csrbin file
 * @param start_row   First row to read (inclusive)
 * @param end_row     Last row to read (exclusive)
 * @param out_global_n  Output: total vertices in graph
 * @return            CSR_info for the requested partition
 */
CSR_info read_csr_bin_part(const char *fname, int start_row, int end_row,
                           int *out_global_n);

#endif /* MAT_TO_CSR_H */
