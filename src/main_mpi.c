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
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided < MPI_THREAD_MULTIPLE) {
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
      printf("  --convert-only      Convert .mat to .csrbin and exit\n");
    }
    MPI_Finalize();
    return 0;
  }

  // Check for --convert-only flag
  int convert_only = 0;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--convert-only") == 0) {
      convert_only = 1;
      break;
    }
  }

  if (convert_only) {
    if (rank == 0) {
      const char *fname =
          (argc >= 2 && argv[1][0] != '-') ? argv[1] : "testmat.mat";
      printf("Running in conversion mode for %s...\n", fname);

      // Load and convert (which implicitly saves cache)
      CSR_info csr = mat_to_csr((char *)fname);

      if (csr.row_ptr) {
        printf("Conversion successful.\n");
        clean_csr_vectors(csr);
      } else {
        fprintf(stderr, "Conversion failed.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
  }

  // Check loading mode: 0 = Original (Rank 0 loads all), 1 = Distributed
  // Edge-Balanced
  char *env_dist = getenv("BFS_DISTRIBUTED");
  int use_distributed = (env_dist && atoi(env_dist) == 1) ? 1 : 0;

  MPI_Graph g; // Declare here so accessible in both scopes
  int n = 0;

  // ==========================================
  // PATH 1: Distributed Loading (Edge Balanced)
  // ==========================================
  if (use_distributed) {
    if (rank == 0)
      printf("Using Distributed Graph Loading (Edge Balanced)...\n");

    const char *fname = (argc >= 2) ? argv[1] : "testmat.mat";
    char bin_name[512];
    snprintf(bin_name, sizeof(bin_name), "%s.csrbin", fname);

    // 1. Check if binary cache exists (Rank 0 check)
    int cache_exists = 0;
    if (rank == 0) {
      FILE *f = fopen(bin_name, "rb");
      if (f) {
        cache_exists = 1;
        fclose(f);
      } else {
        // If not exists, create it using standard loader
        printf("Cache %s not found. Creating it...\n", bin_name);
        CSR_info temp_csr = mat_to_csr((char *)fname);
        if (temp_csr.row_ptr) {
          cache_exists = 1;
          clean_csr_vectors(temp_csr);
        } else {
          fprintf(stderr, "Rank 0: Failed to create/load matrix.\n");
        }
      }
    }

    check_mpi(MPI_Bcast(&cache_exists, 1, MPI_INT, 0, MPI_COMM_WORLD),
              "MPI_Bcast cache_exists");

    if (!cache_exists) {
      MPI_Finalize();
      return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // 1.5 Calculate Partition (Edge-Balanced)
    int *vtx_dist = malloc((size + 1) * sizeof(int));
    if (rank == 0) {
      int *dist = compute_balanced_partition(bin_name, size, NULL);
      if (!dist) {
        fprintf(stderr, "Rank 0: Failed to compute balanced partition.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      memcpy(vtx_dist, dist, (size + 1) * sizeof(int));
      free(dist);
      printf("Balanced Partition:\n");
      for (int i = 0; i < size; i++) {
        printf("  Rank %d: Rows %d - %d\n", i, vtx_dist[i], vtx_dist[i + 1]);
      }
    }

    check_mpi(MPI_Bcast(vtx_dist, size + 1, MPI_INT, 0, MPI_COMM_WORLD),
              "MPI_Bcast vtx_dist");

    // 2. Everyone reads their part
    int global_n = 0;
    int my_start = vtx_dist[rank];
    int my_end = vtx_dist[rank + 1];

    CSR_info local_csr =
        read_csr_bin_part(bin_name, my_start, my_end, &global_n);

    if (!local_csr.row_ptr) {
      fprintf(stderr, "Rank %d: Failed to read partial CSR.\n", rank);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    n = global_n;
    mpi_init_graph_distributed(&g, n, local_csr.nrows, local_csr.row_ptr,
                               local_csr.col_idx, local_csr.nzcount, vtx_dist);
    free(vtx_dist);
  }
  // ==========================================
  // PATH 2: Original Loading (Rank 0 Central)
  // ==========================================
  else {
    if (rank == 0)
      printf("Using Original Graph Loading (Rank 0 Central)...\n");

    CSR_info csr = {0};
    if (rank == 0) {
      const char *fname = (argc >= 2) ? argv[1] : "testmat.mat";
      csr = mat_to_csr((char *)fname);
      if (!csr.row_ptr || !csr.col_idx) {
        fprintf(stderr, "Error: Failed to load matrix %s\n", fname);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      n = csr.nrows;
      printf("Loaded: %d vertices, %lld edges\n", csr.nrows, csr.nzcount);
      fflush(stdout);
    }

    check_mpi(MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast n");

    mpi_init_graph(&g, n, csr.row_ptr, csr.col_idx);

    if (rank == 0) {
      clean_csr_vectors(csr);
    }
  }

  // Run distributed connected components
  double t0 = MPI_Wtime();
  int iters = 0;
  char *env_async = getenv("BFS_ASYNC");
  if (env_async && atoi(env_async) == 1) {
    if (rank == 0)
      printf("Running ASYNC version...\n");
    iters = mpi_coloring_cc_async(&g);
  } else {
    iters = mpi_coloring_cc(&g);
  }
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
  }

  mpi_cleanup_graph(&g);
  MPI_Finalize();
  return 0;
}
