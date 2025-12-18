#ifndef BFS_MPI_H
#define BFS_MPI_H

#include <mpi.h>
#include <stdint.h>
#include <string.h>

// ============================================================================
// Core Graph Data Structure
// ============================================================================

typedef struct {
  int rank;                 // MPI process rank
  int size;                 // Total number of MPI processes
  int local_n;              // Number of local vertices
  int global_n;             // Total vertices in the graph
  long long *local_row_ptr; // Local CSR row pointers (size: local_n + 1)
  int *local_col_idx;       // Local CSR column indices (size: local_nnz)
  int *local_labels;        // Local component labels (size: local_n)
  int *vtx_dist; // Distribution of vertices across processes (size: size + 1)
  long long local_nnz; // Number of local edges

  // Optimization: In-Place Partitioned Adjacency List
  // For each vertex i, local_col_idx[local_row_ptr[i]...split_indices[i]-1]
  // contains internal edges.
  // local_col_idx[split_indices[i]...local_row_ptr[i+1]-1] contains remote
  // edges.
  long long *split_indices;
  long long internal_nnz;
  long long remote_nnz;
} MPI_Graph;

// ============================================================================
// Communication Data Structures
// ============================================================================

typedef struct {
  int *boundary_indices;
  int boundary_count;
  int *send_counts;
  int *recv_counts;
  int *send_displs;
  int *recv_displs;
  int *send_buf;
  int *recv_buf;
  int send_buf_cap;
  int recv_buf_cap;
  MPI_Request *reqs;
} CommInfo;

// Thread Local Storage for Parallel Packing
typedef struct {
  int **buffers; // [rank][data...]
  int *counts;   // [rank]
  int *caps;     // [rank]
} TLS_Buffer;

// Stream Buffer for Streaming BFS
typedef struct {
  int *primary;    // Buffer currently being filled
  int *secondary;  // Buffer currently being sent (or free)
  int count;       // Number of items in primary
  int cap;         // Capacity of each buffer (entries, not bytes)
  MPI_Request req; // Request for the secondary buffer (if in flight)
  int active;      // 1 if a send is currently active (req is valid)
} StreamBuffer;

// Context for Streaming BFS
typedef struct {
  long long total_sent;
  long long total_recv;
} StreamContext;

// TLS for Streaming BFS (Thread-Local Buffers per destination rank)
typedef struct {
  int **buffers; // [size][cap]
  int *counts;   // [size]
  int *caps;     // [size]
} TLS_Stream_Dynamic;

// ============================================================================
// Bitset Helpers (for frontier tracking)
// ============================================================================

#define BITSET_WORD_SIZE (sizeof(uint64_t) * 8)
#define BITSET_WORDS(n) (((n) + BITSET_WORD_SIZE - 1) / BITSET_WORD_SIZE)

static inline void bitset_set(uint64_t *b, int idx) {
  __atomic_fetch_or(&b[idx / BITSET_WORD_SIZE],
                    (1UL << (idx % BITSET_WORD_SIZE)), __ATOMIC_RELAXED);
}

static inline int bitset_test(const uint64_t *b, int idx) {
  return (b[idx / BITSET_WORD_SIZE] & (1UL << (idx % BITSET_WORD_SIZE))) != 0;
}

static inline void bitset_clear(uint64_t *b, size_t num_words) {
  memset(b, 0, num_words * sizeof(uint64_t));
}

// ============================================================================
// Helper to find owner of a global vertex (binary search on vtx_dist)
// ============================================================================

static inline int get_owner(int global_id, const MPI_Graph *g) {
  int l = 0, r = g->size - 1;
  while (l <= r) {
    int m = l + (r - l) / 2;
    if (global_id >= g->vtx_dist[m] && global_id < g->vtx_dist[m + 1]) {
      return m;
    }
    if (global_id < g->vtx_dist[m]) {
      r = m - 1;
    } else {
      l = m + 1;
    }
  }
  return g->size - 1; // Fallback (should be covered)
}

// ============================================================================
// Graph Initialization API
// ============================================================================

// Partition vertices: process p gets vertices [vtx_dist[p], vtx_dist[p+1])
void compute_vertex_distribution(int n, int nprocs, int *vtx_dist);

// Initialize MPI and distribute graph
// row_ptr and col_idx are only valid on rank 0
int mpi_init_graph(MPI_Graph *g, int n, const long long *row_ptr,
                   const int *col_idx);

int mpi_init_graph_distributed(MPI_Graph *g, int global_n, int local_n,
                               long long *local_row_ptr, int *local_col_idx,
                               long long local_nnz, const int *vtx_dist);

// ============================================================================
// Connected Components Algorithms
// ============================================================================

// Main dispatcher (selects algorithm based on environment variables)
int mpi_coloring_cc(MPI_Graph *g);

// Asynchronous label propagation
int mpi_coloring_cc_async(MPI_Graph *g);

// Frontier-Based BFS (O(diameter) iterations)
int mpi_coloring_cc_bfs(MPI_Graph *g);

// Streaming Frontier BFS (overlapped communication)
int mpi_coloring_cc_bfs_streaming(MPI_Graph *g);

// ============================================================================
// Results and Cleanup
// ============================================================================

// Gather final results to rank 0
// global_labels must be allocated on rank 0 (size n)
int mpi_gather_labels(MPI_Graph *g, int *global_labels);

// Cleanup
void mpi_cleanup_graph(MPI_Graph *g);

#endif
