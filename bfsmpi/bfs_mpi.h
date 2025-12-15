#ifndef BFS_MPI_H
#define BFS_MPI_H

#include <mpi.h>

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

// Partition vertices: process p gets vertices [vtx_dist[p], vtx_dist[p+1])
void compute_vertex_distribution(int n, int nprocs, int *vtx_dist);

// Initialize MPI and distribute graph
// row_ptr and col_idx are only valid on rank 0
int mpi_init_graph(MPI_Graph *g, int n, const long long *row_ptr,
                   const int *col_idx);

int mpi_init_graph_distributed(MPI_Graph *g, int global_n, int local_n,
                               long long *local_row_ptr, int *local_col_idx,
                               long long local_nnz, const int *vtx_dist);

// Distributed connected components algorithm
int mpi_coloring_cc(MPI_Graph *g);

// Asynchronous distributed connected components algorithm
int mpi_coloring_cc_async(MPI_Graph *g);

// Gather final results to rank 0
// global_labels must be allocated on rank 0 (size n)
int mpi_gather_labels(MPI_Graph *g, int *global_labels);

// Cleanup
void mpi_cleanup_graph(MPI_Graph *g);

#endif
