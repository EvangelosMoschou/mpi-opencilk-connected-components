#include "bfs_mpi.h"
#include <stdlib.h>

// Partition vertices: process p gets vertices [vtx_dist[p], vtx_dist[p+1])
void compute_vertex_distribution(int n, int nprocs, int *vtx_dist) {
    int base = n / nprocs;
    int remainder = n % nprocs;
    vtx_dist[0] = 0;
    for (int p = 0; p < nprocs; p++) {
        vtx_dist[p+1] = vtx_dist[p] + base + (p < remainder ? 1 : 0);
    }
}
