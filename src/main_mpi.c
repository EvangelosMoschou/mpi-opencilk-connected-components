#define _POSIX_C_SOURCE 200809L
#include "bfsmpi/bfs_mpi.h"
#include "mat_to_csr.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper functions copied from BFScilk.c (modified for 0-based labels)
int count_components(int n, const int *labels) {
  if (n <= 0 || !labels)
    return 0;
  unsigned char *seen = calloc((size_t)(n + 1), 1); // 1-based labels
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

void check_mpi(int err, const char *msg) {
  if (err != MPI_SUCCESS) {
    int len;
    char err_string[MPI_MAX_ERROR_STRING];
    MPI_Error_string(err, err_string, &len);
    fprintf(stderr, "MPI Error in %s: %s\n", msg, err_string);
    MPI_Abort(MPI_COMM_WORLD, err);
  }
}

int main(int argc, char *argv[]) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided < MPI_THREAD_FUNNELED) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
      fprintf(
          stderr,
          "Warning: MPI implementation does not support MPI_THREAD_FUNNELED\n");
  }

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc > 1 &&
      (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
    if (rank == 0) {
      printf("Usage: %s [matrix_file] [options]\n", argv[0]);
      printf("Options:\n");
      printf("  --help, -h          Show this help message\n");
    }
    MPI_Finalize();
    return 0;
  }

  // Only rank 0 reads the matrix file
  CSR_info csr = {0};
  int n = 0;

  if (rank == 0) {
    const char *fname = (argc >= 2) ? argv[1] : "testmat.mat";
    csr = mat_to_csr((char *)fname);
    if (!csr.row_ptr || !csr.col_idx) {
      fprintf(stderr, "Error: Failed to load matrix %s\n", fname);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    n = csr.nrows;
    printf("Loaded: %d vertices, %d edges\n", csr.nrows, csr.nzcount);
    fflush(stdout);
  }

  // Broadcast graph size
  check_mpi(MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast n");

  // Initialize distributed graph structure
  MPI_Graph g;
  mpi_init_graph(&g, n, csr.row_ptr, csr.col_idx);

  // Run distributed connected components
  double t0 = MPI_Wtime();
  int iters = mpi_coloring_cc(&g);
  double t1 = MPI_Wtime();

  // Gather results
  int *global_labels = NULL;
  if (rank == 0) {
    global_labels = malloc(n * sizeof(int));
  }
  mpi_gather_labels(&g, global_labels);

  // Report results from rank 0
  if (rank == 0) {
    int comps = count_components(n, global_labels);
    printf("MPI CC: procs=%d iters=%d time=%.6f s components=%d\n", size, iters,
           t1 - t0, comps);
    fflush(stdout);
    free(global_labels);
    clean_csr_vectors(csr);
  }

  mpi_cleanup_graph(&g);
  MPI_Finalize();
  return 0;
}
