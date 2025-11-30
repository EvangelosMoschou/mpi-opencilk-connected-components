#ifndef BFS_MPI_H
#define BFS_MPI_H

#include <mpi.h>

typedef struct {
  int rank;           // MPI process rank
  int size;           // Total number of MPI processes
  int local_n;        // Number of local vertices
  int global_n;       // Total vertices in the graph
  int *local_row_ptr; // Local CSR row pointers (size: local_n + 1)
  int *local_col_idx; // Local CSR column indices (size: local_nnz)
  int *local_labels;  // Local component labels (size: local_n)
  int *vtx_dist; // Distribution of vertices across processes (size: size + 1)
  int local_nnz; // Number of local edges

  // Optimization: Split adjacency lists
  int *internal_row_ptr;
  int *internal_col_idx;
  int internal_nnz;

  int *remote_row_ptr;
  int *remote_col_idx;
  int *remote_owners;
  int remote_nnz;
} MPI_Graph;

// Partition vertices: process p gets vertices [vtx_dist[p], vtx_dist[p+1])
void compute_vertex_distribution(int n, int nprocs, int *vtx_dist);

// Initialize MPI and distribute graph
// row_ptr and col_idx are only valid on rank 0
int mpi_init_graph(MPI_Graph *g, int n, const int *row_ptr, const int *col_idx);

// Distributed connected components algorithm
int mpi_coloring_cc(MPI_Graph *g);

// Gather final results to rank 0
// global_labels must be allocated on rank 0 (size n)
int mpi_gather_labels(MPI_Graph *g, int *global_labels);

// Cleanup
void mpi_cleanup_graph(MPI_Graph *g);

#endif
