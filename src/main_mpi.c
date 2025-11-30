#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "mat_to_csr.h"
#include "bfsmpi/bfs_mpi.h"

// Helper functions copied from BFScilk.c (modified for 0-based labels)
int count_components(int n, const int *labels) {
    if (n <= 0 || !labels) return 0;
    unsigned char *seen = calloc((size_t)(n + 1), 1); // 1-based labels
    if (!seen) return -1;
    int count = 0;
    for (int i = 0; i < n; ++i) {
        int lbl = labels[i];
        if (lbl < 1 || lbl > n) continue;
        if (!seen[lbl]) { seen[lbl] = 1; ++count; }
    }
    free(seen);
    return count;
}

void symmetrize_csr(int n, int nz, int *row_ptr, int *col_idx, int **new_row_ptr, int **new_col_idx, int *new_nz_out) {
    // 1. Count degrees for transpose
    int *trans_deg = calloc(n, sizeof(int));
    for (int i = 0; i < nz; i++) {
        trans_deg[col_idx[i]]++;
    }
    
    // 2. Allocate new row_ptr
    *new_row_ptr = aligned_alloc_wrapper((n + 1) * sizeof(int));
    if (*new_row_ptr) memset(*new_row_ptr, 0, (n + 1) * sizeof(int));
    
    // 3. Calculate new degrees (original + transpose)
    for (int i = 0; i < n; i++) {
        int orig_deg = row_ptr[i+1] - row_ptr[i];
        (*new_row_ptr)[i+1] = (*new_row_ptr)[i] + orig_deg + trans_deg[i];
    }
    
    int total_nz = (*new_row_ptr)[n];
    *new_col_idx = aligned_alloc_wrapper(total_nz * sizeof(int));
    *new_nz_out = total_nz;
    
    // 4. Fill
    int *offsets = calloc(n, sizeof(int));
    
    // Copy original
    for (int r = 0; r < n; r++) {
        for (int j = row_ptr[r]; j < row_ptr[r+1]; j++) {
            int c = col_idx[j];
            int dest = (*new_row_ptr)[r] + offsets[r]++;
            (*new_col_idx)[dest] = c;
            
            // Add transpose edge c -> r
            int dest_t = (*new_row_ptr)[c] + offsets[c]++;
            (*new_col_idx)[dest_t] = r;
        }
    }
    
    free(trans_deg);
    free(offsets);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Only rank 0 reads the matrix file
    CSR_info csr = {0};
    int n = 0;
    
    if (rank == 0) {
        const char *fname = (argc >= 2) ? argv[1] : "testmat.mat";
        csr = mat_to_csr((char *)fname);
        n = csr.nrows;
        printf("Loaded: %d vertices, %d edges\n", csr.nrows, csr.nzcount);

        // Check for symmetrization flag
        int do_symmetrize = 0;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--symmetrize") == 0 || strcmp(argv[i], "-s") == 0) {
                do_symmetrize = 1;
                break;
            }
        }

        if (do_symmetrize) {
            // Symmetrize graph to ensure undirected connectivity
            int *s_row_ptr, *s_col_idx;
            int s_nz;
            symmetrize_csr(n, csr.nzcount, csr.row_ptr, csr.col_idx, &s_row_ptr, &s_col_idx, &s_nz);
            
            free(csr.row_ptr);
            free(csr.col_idx);
            csr.row_ptr = s_row_ptr;
            csr.col_idx = s_col_idx;
            csr.nzcount = s_nz;
            printf("Symmetrized graph: %d edges\n", s_nz);
        } else {
            printf("Skipping symmetrization (use --symmetrize to enable)\n");
        }
    }
    
    // Broadcast graph size
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
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
        printf("MPI CC: procs=%d iters=%d time=%.6f s components=%d\n",
               size, iters, t1 - t0, comps);
        free(global_labels);
        clean_csr_vectors(csr);
    }
    
    mpi_cleanup_graph(&g);
    MPI_Finalize();
    return 0;
}
