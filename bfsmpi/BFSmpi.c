// Keep the implementation label propagation only, use parts of other algorithms
// But keep the base label propagation. It is asked for the assignment.
#define _POSIX_C_SOURCE 200809L
#include "bfs_mpi.h"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper for aligned allocation (64-byte alignment for AVX-512/Cache lines)
static void *aligned_alloc_wrapper(size_t size) {
  void *ptr = NULL;
  if (posix_memalign(&ptr, 64, size) != 0) {
    fprintf(stderr, "Error: Aligned allocation failed for size %zu\n", size);
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_NO_MEM);
    return NULL;
  }
  return ptr;
}

// Atomic Min Helper
static inline int atomic_min(int *addr, int val) {
  int old = *addr;
  while (val < old) {
    if (__sync_bool_compare_and_swap(addr, old, val))
      return 1;
    old = *addr;
  }
  return 0;
}

// Bitset helpers
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

// Helper to find owner of a global vertex
static inline int get_owner(int global_id, int base, int cutoff,
                            int base_plus_1) {
  if (global_id < cutoff) {
    return global_id / base_plus_1;
  } else {
    return (cutoff / base_plus_1) + (global_id - cutoff) / base;
  }
}

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

static CommInfo comm;

void setup_communication(MPI_Graph *g) {
  // Precompute distribution parameters
  int base = g->global_n / g->size;
  int remainder = g->global_n % g->size;
  int cutoff = remainder * (base + 1);
  int base_plus_1 = base + 1;

  // 1. Count internal and remote edges
  g->internal_nnz = 0;
  g->remote_nnz = 0;
  comm.boundary_count = 0;

  int *temp_boundary = malloc(g->local_n * sizeof(int));

  for (int i = 0; i < g->local_n; i++) {
    int start = g->local_row_ptr[i];
    int end = g->local_row_ptr[i + 1];
    int has_remote = 0;
    for (int j = start; j < end; j++) {
      int neighbor = g->local_col_idx[j];
      if (neighbor >= g->vtx_dist[g->rank] &&
          neighbor < g->vtx_dist[g->rank + 1]) {
        g->internal_nnz++;
      } else {
        g->remote_nnz++;
        has_remote = 1;
      }
    }
    if (has_remote) {
      temp_boundary[comm.boundary_count++] = i;
    }
  }

  comm.boundary_indices = malloc(comm.boundary_count * sizeof(int));
  memcpy(comm.boundary_indices, temp_boundary,
         comm.boundary_count * sizeof(int));
  free(temp_boundary);

  // 2. Allocate split arrays
  g->internal_row_ptr = aligned_alloc_wrapper((g->local_n + 1) * sizeof(int));
  g->internal_col_idx = aligned_alloc_wrapper(g->internal_nnz * sizeof(int));

  g->remote_row_ptr = aligned_alloc_wrapper((g->local_n + 1) * sizeof(int));
  g->remote_col_idx = aligned_alloc_wrapper(g->remote_nnz * sizeof(int));
  g->remote_owners = aligned_alloc_wrapper(g->remote_nnz * sizeof(int));

  // 3. Fill split arrays
  int internal_pos = 0;
  int remote_pos = 0;
  g->internal_row_ptr[0] = 0;
  g->remote_row_ptr[0] = 0;

  for (int i = 0; i < g->local_n; i++) {
    int start = g->local_row_ptr[i];
    int end = g->local_row_ptr[i + 1];

    for (int j = start; j < end; j++) {
      int neighbor = g->local_col_idx[j];
      if (neighbor >= g->vtx_dist[g->rank] &&
          neighbor < g->vtx_dist[g->rank + 1]) {
        g->internal_col_idx[internal_pos++] = neighbor;
      } else {
        g->remote_col_idx[remote_pos] = neighbor;
        // Compute owner
        int owner;
        if (neighbor < cutoff) {
          owner = neighbor / base_plus_1;
        } else {
          owner = (cutoff / base_plus_1) + (neighbor - cutoff) / base;
        }
        g->remote_owners[remote_pos] = owner;
        remote_pos++;
      }
    }
    g->internal_row_ptr[i + 1] = internal_pos;
    g->remote_row_ptr[i + 1] = remote_pos;
  }

  // Free original adjacency lists to save memory
  free(g->local_row_ptr);
  free(g->local_col_idx);
  g->local_row_ptr = NULL;
  g->local_col_idx = NULL;

  // Init comm buffers
  comm.send_counts = calloc(g->size, sizeof(int));
  comm.recv_counts = calloc(g->size, sizeof(int));
  comm.send_displs = calloc(g->size, sizeof(int));
  comm.recv_displs = calloc(g->size, sizeof(int));

  comm.send_buf_cap = 1024 * 1024 * 16; // Start with 16MB buffer
  comm.recv_buf_cap = 1024 * 1024 * 16;
  comm.send_buf = aligned_alloc_wrapper(comm.send_buf_cap * sizeof(int));
  comm.recv_buf = aligned_alloc_wrapper(comm.recv_buf_cap * sizeof(int));
  comm.reqs = malloc(2 * g->size * sizeof(MPI_Request));
}

void cleanup_communication() {
  free(comm.boundary_indices);
  free(comm.send_counts);
  free(comm.recv_counts);
  free(comm.send_displs);
  free(comm.recv_displs);
  free(comm.send_buf);
  free(comm.recv_buf);
  free(comm.reqs);
}

int mpi_coloring_cc(MPI_Graph *g) {
  setup_communication(g);

  int rank = g->rank;
  int size = g->size;
  int local_n = g->local_n;
  int *labels = g->local_labels;
  int my_offset = g->vtx_dist[rank];

  // Inform compiler about alignment
  labels = __builtin_assume_aligned(labels, 64);
  g->local_row_ptr = __builtin_assume_aligned(g->local_row_ptr, 64);
  g->local_col_idx = __builtin_assume_aligned(g->local_col_idx, 64);
  comm.send_buf = __builtin_assume_aligned(comm.send_buf, 64);
  comm.recv_buf = __builtin_assume_aligned(comm.recv_buf, 64);

  // Precompute distribution parameters for O(1) owner lookup
  int base = g->global_n / size;
  int remainder = g->global_n % size;
  int cutoff = remainder * (base + 1);
  int base_plus_1 = base + 1;

  // Initialize labels (1-based)
  cilk_for(int i = 0; i < local_n; i++) { labels[i] = my_offset + i + 1; }

  // Frontier arrays to track changed vertices (Bitsets)
  size_t frontier_words = BITSET_WORDS(local_n);
  uint64_t *changed = aligned_alloc_wrapper(frontier_words * sizeof(uint64_t));
  uint64_t *next_changed =
      aligned_alloc_wrapper(frontier_words * sizeof(uint64_t));

  // Initially all vertices are active
  memset(changed, 0xFF, frontier_words * sizeof(uint64_t));
  // Clear padding bits in the last word if necessary
  if (local_n % BITSET_WORD_SIZE != 0) {
    uint64_t mask = (1UL << (local_n % BITSET_WORD_SIZE)) - 1;
    changed[frontier_words - 1] &= mask;
  }

  int global_changed = 1;
  int iterations = 0;

  // Buffers for packing
  int **proc_buffers = malloc(size * sizeof(int *));
  int *proc_caps = malloc(size * sizeof(int));
  int *proc_counts = malloc(size * sizeof(int));
  for (int p = 0; p < size; p++) {
    proc_caps[p] = 1024 * 1024; // Start with 1MB per proc
    proc_buffers[p] = aligned_alloc_wrapper(proc_caps[p] * sizeof(int));
    proc_counts[p] = 0;
  }

  while (global_changed) {
    iterations++;
    int local_changed = 0;

    // Reset next_changed
    bitset_clear(next_changed, frontier_words);

    // 1. Local Propagation
    cilk_for(int i = 0; i < local_n; i++) {
      if (bitset_test(changed, i)) {
        int val = labels[i];

        // Path Compression
        while (val > my_offset && val <= my_offset + local_n) {
          int parent = labels[val - my_offset - 1];
          if (parent < val)
            val = parent;
          else
            break;
        }

        int start = g->internal_row_ptr[i];
        int end = g->internal_row_ptr[i + 1];
        for (int j = start; j < end; j++) {
          int neighbor = g->internal_col_idx[j];
          // No range check needed for internal edges
          if (atomic_min(&labels[neighbor - my_offset], val)) {
            bitset_set(next_changed, neighbor - my_offset);
          }
        }
      }
    }

    // 2. Pack Messages
    for (int p = 0; p < size; p++)
      proc_counts[p] = 0;

    for (int i = 0; i < comm.boundary_count; i++) {
      int u = comm.boundary_indices[i];
      if (bitset_test(changed, u)) {
        int val = labels[u];

        // Path Compression
        while (val > my_offset && val <= my_offset + local_n) {
          int parent = labels[val - my_offset - 1];
          if (parent < val)
            val = parent;
          else
            break;
        }

        int start = g->remote_row_ptr[u];
        int end = g->remote_row_ptr[u + 1];
        for (int j = start; j < end; j++) {
          int neighbor = g->remote_col_idx[j];
          int owner = g->remote_owners[j];

          if (proc_counts[owner] + 2 > proc_caps[owner]) {
            proc_caps[owner] *= 2;
            int *new_buf =
                aligned_alloc_wrapper(proc_caps[owner] * sizeof(int));
            memcpy(new_buf, proc_buffers[owner],
                   proc_counts[owner] * sizeof(int));
            free(proc_buffers[owner]);
            proc_buffers[owner] = new_buf;
          }
          proc_buffers[owner][proc_counts[owner]++] = neighbor;
          proc_buffers[owner][proc_counts[owner]++] = val;
        }
      }
    }

    // 3. Exchange
    int err = MPI_Alltoall(proc_counts, 1, MPI_INT, comm.recv_counts, 1,
                           MPI_INT, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS)
      MPI_Abort(MPI_COMM_WORLD, err);

    int total_recv = 0;
    for (int p = 0; p < size; p++) {
      comm.recv_displs[p] = total_recv;
      total_recv += comm.recv_counts[p];
    }

    if (total_recv > comm.recv_buf_cap) {
      comm.recv_buf_cap = total_recv + 1024;
      int *new_buf = aligned_alloc_wrapper(comm.recv_buf_cap * sizeof(int));

      free(comm.recv_buf);
      comm.recv_buf = new_buf;
    }

    int nreqs = 0;

    for (int p = 0; p < size; p++) {
      if (comm.recv_counts[p] > 0) {
        err =
            MPI_Irecv(comm.recv_buf + comm.recv_displs[p], comm.recv_counts[p],
                      MPI_INT, p, 0, MPI_COMM_WORLD, &comm.reqs[nreqs++]);
        if (err != MPI_SUCCESS)
          MPI_Abort(MPI_COMM_WORLD, err);
      }
    }
    for (int p = 0; p < size; p++) {
      if (proc_counts[p] > 0) {
        err = MPI_Isend(proc_buffers[p], proc_counts[p], MPI_INT, p, 0,
                        MPI_COMM_WORLD, &comm.reqs[nreqs++]);
        if (err != MPI_SUCCESS)
          MPI_Abort(MPI_COMM_WORLD, err);
      }
    }

    err = MPI_Waitall(nreqs, comm.reqs, MPI_STATUSES_IGNORE);
    if (err != MPI_SUCCESS)
      MPI_Abort(MPI_COMM_WORLD, err);

    // 4. Process Received
    for (int p = 0; p < size; p++) {
      int count = comm.recv_counts[p];
      int *buf = comm.recv_buf + comm.recv_displs[p];
      for (int k = 0; k < count; k += 2) {
        int target = buf[k];
        int val = buf[k + 1];
        if (target >= my_offset && target < my_offset + local_n) {
          if (atomic_min(&labels[target - my_offset], val)) {
            bitset_set(next_changed, target - my_offset);
          }
        }
      }
    }

    // 5. Termination
    // Swap changed and next_changed
    uint64_t *tmp = changed;
    changed = next_changed;
    next_changed = tmp;

    // Check if any local changes occurred
    local_changed = 0;
    for (size_t i = 0; i < frontier_words; i++) {
      if (changed[i] != 0) {
        local_changed = 1;
        break;
      }
    }

    MPI_Allreduce(&local_changed, &global_changed, 1, MPI_INT, MPI_LOR,
                  MPI_COMM_WORLD);
  }

  for (int p = 0; p < size; p++)
    free(proc_buffers[p]);
  free(proc_buffers);
  free(proc_caps);
  free(proc_counts);
  free(changed);
  free(next_changed);
  cleanup_communication();
  return iterations;
}

void mpi_cleanup_graph(MPI_Graph *g) {
  // local_row_ptr and local_col_idx are freed in setup_communication
  free(g->local_labels);
  free(g->vtx_dist);

  // Free split arrays
  free(g->internal_row_ptr);
  free(g->internal_col_idx);
  free(g->remote_row_ptr);
  free(g->remote_col_idx);
  free(g->remote_owners);
}

int mpi_gather_labels(MPI_Graph *g, int *global_labels) {
  int *recv_counts = NULL;
  int *displs = NULL;
  if (g->rank == 0) {
    recv_counts = malloc(g->size * sizeof(int));
    displs = malloc(g->size * sizeof(int));
    for (int p = 0; p < g->size; p++) {
      recv_counts[p] = g->vtx_dist[p + 1] - g->vtx_dist[p];
      displs[p] = g->vtx_dist[p];
    }
  }
  MPI_Gatherv(g->local_labels, g->local_n, MPI_INT, global_labels, recv_counts,
              displs, MPI_INT, 0, MPI_COMM_WORLD);
  if (g->rank == 0) {
    free(recv_counts);
    free(displs);
  }
  return 0;
}

int mpi_init_graph(MPI_Graph *g, int n, const int *row_ptr,
                   const int *col_idx) {
  MPI_Comm_rank(MPI_COMM_WORLD, &g->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g->size);
  g->global_n = n;

  g->vtx_dist = malloc((g->size + 1) * sizeof(int));
  // Re-implement compute_vertex_distribution inline or call it if available
  int base = n / g->size;
  int remainder = n % g->size;
  g->vtx_dist[0] = 0;
  for (int p = 0; p < g->size; p++) {
    g->vtx_dist[p + 1] = g->vtx_dist[p] + base + (p < remainder ? 1 : 0);
  }

  g->local_n = g->vtx_dist[g->rank + 1] - g->vtx_dist[g->rank];

  g->local_row_ptr = aligned_alloc_wrapper((g->local_n + 1) * sizeof(int));

  if (g->rank == 0) {
    for (int p = 1; p < g->size; p++) {
      int p_start = g->vtx_dist[p];
      int p_count = (g->vtx_dist[p + 1] - g->vtx_dist[p]) + 1;
      MPI_Send(&row_ptr[p_start], p_count, MPI_INT, p, 0, MPI_COMM_WORLD);
    }
    memcpy(g->local_row_ptr, row_ptr, (g->local_n + 1) * sizeof(int));
  } else {
    MPI_Recv(g->local_row_ptr, g->local_n + 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

  int global_start_idx = g->local_row_ptr[0];
  for (int i = 0; i <= g->local_n; i++) {
    g->local_row_ptr[i] -= global_start_idx;
  }

  g->local_nnz = g->local_row_ptr[g->local_n];
  g->local_col_idx = aligned_alloc_wrapper(g->local_nnz * sizeof(int));
  g->local_labels = aligned_alloc_wrapper(g->local_n * sizeof(int));

  int *sendcounts = NULL;
  int *displs = NULL;

  if (g->rank == 0) {
    sendcounts = malloc(g->size * sizeof(int));
    displs = malloc(g->size * sizeof(int));
    for (int p = 0; p < g->size; p++) {
      int p_start = g->vtx_dist[p];
      int p_end = g->vtx_dist[p + 1];
      int start_idx = row_ptr[p_start];
      int end_idx = row_ptr[p_end];
      sendcounts[p] = end_idx - start_idx;
      displs[p] = start_idx;
    }
  }

  MPI_Scatterv(col_idx, sendcounts, displs, MPI_INT, g->local_col_idx,
               g->local_nnz, MPI_INT, 0, MPI_COMM_WORLD);

  if (g->rank == 0) {
    free(sendcounts);
    free(displs);
  }

  return 0;
}
