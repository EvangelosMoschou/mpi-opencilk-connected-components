/*
 * main.c - Program entry point for connected components computation
 *
 * This program loads a graph from a MATLAB .mat file, computes its connected
 * components using the Cilk parallel algorithm, and reports timing
 * and component information.
 *
 * Usage: ./main <matrix_file.mat>
 */

#define _POSIX_C_SOURCE 200809L
#include "bfscilk/bfs.h"
#include "mat_to_csr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 * now_seconds - Get current time in seconds
 *
 * Returns: current time as a double-precision floating point value
 */
static double now_seconds(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char *argv[]) {

  /* Default matrix file if none is specified on command line */
  const char *fname = "testmat2.mat";
  int convert_only = 0;

  // Simple argument parsing
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--convert-only") == 0) {
      convert_only = 1;
    } else if (argv[i][0] != '-') {
      fname = argv[i];
    }
  }

  /* We'll use the OpenCilk implementation only */

  /* Load the graph from the MATLAB file into CSR format */
  CSR_info csr = mat_to_csr((char *)fname);

  if (convert_only) {
    if (csr.row_ptr) {
      printf("Conversion successful for %s.\n", fname);
      clean_csr_vectors(csr);
      return 0;
    } else {
      fprintf(stderr, "Conversion failed for %s.\n", fname);
      return 1;
    }
  }

  int rows = csr.nrows;
  int nz = csr.nzcount;
  printf("Loaded matrix:\n number of rows:%d,\n number of nonzeros:%d\n", rows,
         nz);

  /* Allocate space for component labels - size based on number of rows, not
   * nonzeros */
  int *labels = NULL;
  if (posix_memalign((void **)&labels, 64, (size_t)rows * sizeof(int)) != 0) {
    fprintf(stderr, "OOM allocating labels\n");
    clean_csr_vectors(csr);
    return 3;
  }

  double t0 = now_seconds();
  int iters = 0;

  /* Use the Cilk coloring implementation */
  printf("Using Cilk coloring implementation\n");
  iters = coloring_cc(rows, csr.row_ptr, csr.col_idx, labels, 0);

  double t1 = now_seconds();

  // Optimization: Free the graph immediately after BFS to save memory
  // before the count_components phase (which allocates 'seen' array).
  clean_csr_vectors(csr);

  /* Count and display results */
  int comps = count_components(rows, labels);
  printf("Coloring CC: iterations=%d time=%.6f s components=%d\n", iters,
         t1 - t0, comps);
  if (comps > 0) {
    printf("First 20 labels (or fewer):\n");
    print_sample_labels(rows, labels, 20);
  }

  /* Clean up allocated memory */
  free(labels);

  return 0;
}
