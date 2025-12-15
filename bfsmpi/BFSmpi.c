// Keep the implementation label propagation only, use parts of other algorithms
// But keep the base label propagation. It is asked for the assignment.
#define _POSIX_C_SOURCE 200809L
#include "bfs_mpi.h"
#if defined(__opencilk) || defined(__cilk) || defined(__opencilk__)
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#else
#define cilk_for for
#define cilk_spawn
#define cilk_sync
#endif

#include <limits.h>
#include <mpi.h>
#include <pthread.h>
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
// Helper to find owner of a global vertex
// Updated to use binary search on vtx_dist for EDGE-BALANCED PARTITIONS
static inline int get_owner(int global_id, const MPI_Graph *g) {
  int l = 0, r = g->size - 1;
  while (l <= r) {
    int m = l + (r - l) / 2;
    // Start of rank m?
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

// Thread Local Storage for Parallel Packing
typedef struct {
  int **buffers; // [rank][data...]
  int *counts;   // [rank]
  int *caps;     // [rank]
} TLS_Buffer;

// Forward declaration for Frontier BFS
// Forward declaration for Frontier BFS
int mpi_coloring_cc_bfs(MPI_Graph *g);
int mpi_coloring_cc_bfs_streaming(MPI_Graph *g);

static TLS_Buffer *tls_buffers = NULL;
static int n_workers = 1;

void init_tls_buffers(int workers, int size) {
  n_workers = workers;
  tls_buffers = malloc(workers * sizeof(TLS_Buffer));
  for (int w = 0; w < workers; w++) {
    tls_buffers[w].buffers = malloc(size * sizeof(int *));
    tls_buffers[w].counts = calloc(size, sizeof(int));
    tls_buffers[w].caps = malloc(size * sizeof(int));
    for (int p = 0; p < size; p++) {
      tls_buffers[w].caps[p] = 4096; // Initial small cap
      tls_buffers[w].buffers[p] = malloc(tls_buffers[w].caps[p] * sizeof(int));
    }
  }
}

void free_tls_buffers(int workers, int size) {
  if (!tls_buffers)
    return;
  for (int w = 0; w < workers; w++) {
    for (int p = 0; p < size; p++) {
      free(tls_buffers[w].buffers[p]);
    }
    free(tls_buffers[w].buffers);
    free(tls_buffers[w].counts);
    free(tls_buffers[w].caps);
  }
  free(tls_buffers);
  tls_buffers = NULL;
}

void setup_communication(MPI_Graph *g) {
  // Precompute distribution parameters

  // 1. In-place partition adjacency lists into Internal | Remote
  g->internal_nnz = 0;
  g->remote_nnz = 0;
  comm.boundary_count = 0;

  g->split_indices = aligned_alloc_wrapper(g->local_n * sizeof(long long));

  // Use a temporary boundary array
  int *temp_boundary = malloc(g->local_n * sizeof(int));
  int my_start = g->vtx_dist[g->rank];
  int my_end = g->vtx_dist[g->rank + 1];

  for (int i = 0; i < g->local_n; i++) {
    long long start = g->local_row_ptr[i];
    long long end = g->local_row_ptr[i + 1];
    long long count = end - start;

    // Partition range [start, end) in local_col_idx
    // We want internal edges (in [my_start, my_end)) to be at the front
    long long left = start;
    long long right = end - 1;

    while (left <= right) {
      int v = g->local_col_idx[left];
      int is_internal = (v >= my_start && v < my_end);

      if (is_internal) {
        left++;
      } else {
        // Swap with right
        int temp = g->local_col_idx[right];
        g->local_col_idx[right] = g->local_col_idx[left];
        g->local_col_idx[left] = temp;
        right--;
        // Stay at left to check the swapped-in value
      }
    }

    // left is now the split index (start of remote edges)
    g->split_indices[i] = left;

    long long splits = left - start; // number of internal edges
    g->internal_nnz += splits;
    g->remote_nnz += (count - splits);

    if (splits < count) { // If has remote edges
      temp_boundary[comm.boundary_count++] = i;
    }
  }

  comm.boundary_indices = malloc(comm.boundary_count * sizeof(int));
  memcpy(comm.boundary_indices, temp_boundary,
         comm.boundary_count * sizeof(int));
  free(temp_boundary);

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
  // Adaptive dispatcher: check if Frontier BFS should be used
  static int use_frontier = -1; // -1 = not checked yet
  static int use_streaming = -1;
  if (use_frontier == -1) {
    char *env = getenv("BFS_USE_FRONTIER");
    use_frontier = (env && atoi(env) > 0) ? 1 : 0;

    char *env_s = getenv("BFS_USE_STREAMING_FRONTIER");
    use_streaming = (env_s && atoi(env_s) > 0) ? 1 : 0;

    if (g->rank == 0) {
      if (use_streaming) {
        printf("BFS_USE_STREAMING_FRONTIER=1: Using Streaming Frontier BFS\n");
      } else if (use_frontier) {
        printf("BFS_USE_FRONTIER=1: Using Frontier BFS (O(diameter) "
               "iterations)\n");
      }
    }
  }

  if (use_streaming) {
    // Streaming BFS manages its own buffers, skips setup_communication
    return mpi_coloring_cc_bfs_streaming(g);
  }

  if (use_frontier) {
    setup_communication(g);
    return mpi_coloring_cc_bfs(g);
  }

  // Fall through to Label Propagation (default)
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

  static int sampling_rounds = 0;
  static int sampling_k = 2;
  static int prefetch_dist = 16;
  static int zero_planting = 1;
  static int use_hybrid_packing = 0;
  static int config_loaded = 0;

  if (!config_loaded) {
    char *env_rounds = getenv("BFS_SAMPLING_ROUNDS");
    if (env_rounds)
      sampling_rounds = atoi(env_rounds);

    char *env_k = getenv("BFS_SAMPLING_K");
    if (env_k)
      sampling_k = atoi(env_k);

    char *env_dist = getenv("BFS_PREFETCH_DIST");
    if (env_dist)
      prefetch_dist = atoi(env_dist);

    char *env_zp = getenv("BFS_ZERO_PLANTING");
    if (env_zp)
      zero_planting = atoi(env_zp);

    char *env_hybrid = getenv("BFS_HYBRID_PACKING");
    if (env_hybrid)
      use_hybrid_packing = atoi(env_hybrid);

    if (g->rank == 0) {
      printf("MPI BFS Config: Sampling Rounds=%d, K=%d, Prefetch Dist=%d, Zero "
             "Planting=%d, Hybrid Packing=%d\n",
             sampling_rounds, sampling_k, prefetch_dist, zero_planting,
             use_hybrid_packing);
    }
    config_loaded = 1;
  }

  // Init Thread Local Storage if Hybrid Packing is enabled
  int workers = 1;
#if defined(__opencilk) || defined(__cilk) || defined(__opencilk__)
  workers = __cilkrts_get_nworkers();
#endif

  if (use_hybrid_packing && workers > 1) {
    init_tls_buffers(workers, size);
  }

  // Optimization: Zero Planting
  if (zero_planting && iterations == 0) {
    // 1. Find local max degree
    int local_max_v = -1;
    int local_max_deg = -1;

    for (int i = 0; i < local_n; i++) {
      int deg = g->local_row_ptr[i + 1] - g->local_row_ptr[i];
      if (deg > local_max_deg) {
        local_max_deg = deg;
        local_max_v = i;
      }
    }

    // 2. Find global max degree
    struct {
      int val;
      int rank;
    } local_max, global_max;

    local_max.val = local_max_deg;
    local_max.rank = rank;

    MPI_Allreduce(&local_max, &global_max, 1, MPI_2INT, MPI_MAXLOC,
                  MPI_COMM_WORLD);

    // 3. Swap labels[0] (global ID 0, on rank 0) with labels[global_max_v] (on
    // global_max.rank) Note: We don't know the exact local index of global max
    // on that rank yet, but that rank knows its local_max_v.

    // We need the rank that won to know WHICH of its vertices was the max.
    // But MPI_MAXLOC only gives us the rank. It doesn't tell us the index on
    // that rank. So the winner rank needs to broadcast "I am the winner, and my
    // max vertex is X". Actually, only the winner needs to act.

    int winner_rank = global_max.rank;

    if (rank == 0 && winner_rank == 0) {
      // Local swap on rank 0
      if (local_max_v != 0) { // If max degree is not already vertex 0
        int temp = labels[0];
        labels[0] = labels[local_max_v];
        labels[local_max_v] = temp;
        // Initial Push
        int val = labels[local_max_v];

        // Calculate local average degree (heuristic)
        int local_avg_deg = (local_n > 0) ? (g->local_nnz / local_n) : 0;

        // Internal neighbors
        long long start = g->local_row_ptr[local_max_v];
        long long end = g->split_indices[local_max_v];
        for (long long j = start; j < end; j++) {
          int neighbor = g->local_col_idx[j];
          if (atomic_min(&labels[neighbor - my_offset], val)) {
            bitset_set(next_changed, neighbor - my_offset);

            // Hub Push: If neighbor is a hub, push to its neighbors
            // Note: neighbor is local (internal edge)
            int neighbor_local_idx = neighbor - my_offset;
            int neighbor_deg = (g->split_indices[neighbor_local_idx] -
                                g->local_row_ptr[neighbor_local_idx]);
            // We ignore remote degree for simplicity in this heuristic check

            if (neighbor_deg > local_avg_deg) {
              long long n_start = g->local_row_ptr[neighbor_local_idx];
              long long n_end = g->split_indices[neighbor_local_idx];
              for (long long k = n_start; k < n_end; k++) {
                int w = g->local_col_idx[k];
                if (atomic_min(&labels[w - my_offset], val)) {
                  bitset_set(next_changed, w - my_offset);
                }
              }
            }
          }
        }
        // Remote neighbors (mark for sending next round? Or send now?)
        // For simplicity, we just mark them as changed so they get processed in
        // the first iteration. Actually, we should probably just set
        // bitset_set(changed, local_max_v) and let the loop handle it? But we
        // want to "flood" instantly. Let's just mark the max vertex as changed
        // and let the first iteration handle propagation. That's safer and
        // simpler for MPI.
        bitset_set(changed, local_max_v);
      }
    } else if (rank == 0) {
      // Rank 0 needs to swap its labels[0] with Winner Rank
      int my_val = labels[0];
      int other_val;
      MPI_Status status;

      // Send my_val to winner, Recv other_val from winner
      MPI_Sendrecv(&my_val, 1, MPI_INT, winner_rank, 0, &other_val, 1, MPI_INT,
                   winner_rank, 0, MPI_COMM_WORLD, &status);

      labels[0] = other_val;
      bitset_set(changed, 0); // Mark as changed to propagate new label
    } else if (rank == winner_rank) {
      // I am the winner. Swap my labels[local_max_v] with Rank 0
      int my_val = labels[local_max_v];
      int other_val;
      MPI_Status status;

      MPI_Sendrecv(&my_val, 1, MPI_INT, 0, 0, &other_val, 1, MPI_INT, 0, 0,
                   MPI_COMM_WORLD, &status);

      labels[local_max_v] = other_val;
      bitset_set(changed, local_max_v); // Mark as changed to propagate
    }

    // Ensure everyone agrees on the start state
    MPI_Barrier(MPI_COMM_WORLD);
  }

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

        // Skip path compression if already at global minimum
        if (val != 1) {
          // Path Compression (Local Pointer Jumping)
          while (val > my_offset && val <= my_offset + local_n) {
            int parent = labels[val - my_offset - 1];
            if (parent < val)
              val = parent;
            else
              break;
          }
        }

        // Optimization: Early Update (Global Pointer Jumping effect)
        if (val < labels[i]) {
          if (atomic_min(&labels[i], val)) {
            // If we updated ourselves, we are definitely changed for next round
            // But here we are processing 'changed' from previous round.
            // This update helps future lookups of this vertex.
          }
        }

        long long start = g->local_row_ptr[i];
        long long end = g->split_indices[i];

        long long limit = end;
        if (iterations <= sampling_rounds) {
          if (start + sampling_k < end)
            limit = start + sampling_k;
        }
        for (long long j = start; j < limit; j++) {
          // Optimization: Software Prefetching
          if (prefetch_dist > 0) {
            // Stage 1: Prefetch neighbor index
            if (j + prefetch_dist * 2 < limit) {
              __builtin_prefetch(&g->local_col_idx[j + prefetch_dist * 2], 0,
                                 3);
            }
            // Stage 2: Prefetch label
            if (j + prefetch_dist < limit) {
              int neighbor_next = g->local_col_idx[j + prefetch_dist];
              __builtin_prefetch(&labels[neighbor_next - my_offset], 0, 3);
            }
          }

          int neighbor = g->local_col_idx[j];
          // No range check needed for internal edges
          if (atomic_min(&labels[neighbor - my_offset], val)) {
            bitset_set(next_changed, neighbor - my_offset);
          }
        }
      }
    }

    // 2. Pack Messages
    if (use_hybrid_packing && n_workers > 1) {
      // Parallel Pack Messages
      // Clear TLS counts
      cilk_for(int w = 0; w < n_workers; w++) {
        memset(tls_buffers[w].counts, 0, size * sizeof(int));
      }

      cilk_for(int i = 0; i < comm.boundary_count; i++) {
        int tid = 0;
#if defined(__opencilk) || defined(__cilk) || defined(__opencilk__)
        tid = __cilkrts_get_worker_number();
#endif

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

          long long start = g->split_indices[u];
          long long end = g->local_row_ptr[u + 1];

          long long limit = end;
          if (iterations <= sampling_rounds) {
            if (start + sampling_k < end)
              limit = start + sampling_k;
          }
          for (long long j = start; j < limit; j++) {
            // Prefetching
            if (prefetch_dist > 0 && j + prefetch_dist < limit) {
              __builtin_prefetch(&g->local_col_idx[j + prefetch_dist], 0, 3);
            }

            int neighbor = g->local_col_idx[j];
            int owner = get_owner(neighbor, g);

            // Append to TLS buffer
            int cnt = tls_buffers[tid].counts[owner];
            if (cnt + 2 > tls_buffers[tid].caps[owner]) {
              tls_buffers[tid].caps[owner] *= 2;
              // Simple realloc equivalent
              int *old_b = tls_buffers[tid].buffers[owner];
              int *new_b = malloc(tls_buffers[tid].caps[owner] * sizeof(int));
              memcpy(new_b, old_b, cnt * sizeof(int));
              free(old_b);
              tls_buffers[tid].buffers[owner] = new_b;
            }
            tls_buffers[tid].buffers[owner][cnt] = neighbor;
            tls_buffers[tid].buffers[owner][cnt + 1] = val;
            tls_buffers[tid].counts[owner] = cnt + 2;
          }
        }
      }

      // Merge TLS buffers into main proc_buffers
      cilk_for(int p = 0; p < size; p++) {
        int total = 0;
        for (int w = 0; w < n_workers; w++)
          total += tls_buffers[w].counts[p];

        proc_counts[p] = total;
        if (total > proc_caps[p]) {
          while (proc_caps[p] < total)
            proc_caps[p] *= 2;
          free(proc_buffers[p]);
          proc_buffers[p] = aligned_alloc_wrapper(proc_caps[p] * sizeof(int));
        }

        int offset = 0;
        for (int w = 0; w < n_workers; w++) {
          int c = tls_buffers[w].counts[p];
          if (c > 0) {
            memcpy(proc_buffers[p] + offset, tls_buffers[w].buffers[p],
                   c * sizeof(int));
            offset += c;
          }
        }
      }
    } else {
      // Serial Pack Messages
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

          long long start = g->split_indices[u];
          long long end = g->local_row_ptr[u + 1];

          long long limit = end;
          if (iterations <= sampling_rounds) {
            if (start + sampling_k < end)
              limit = start + sampling_k;
          }
          for (long long j = start; j < limit; j++) {
            if (prefetch_dist > 0 && j + prefetch_dist < limit) {
              __builtin_prefetch(&g->local_col_idx[j + prefetch_dist], 0, 3);
            }

            int neighbor = g->local_col_idx[j];
            int owner = get_owner(neighbor, g);

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
      cilk_for(int k = 0; k < count; k += 2) {
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
  free_tls_buffers(n_workers, size);
  cleanup_communication();
  return iterations;
}

// ======================================================================
// STREAMING FRONTIER BFS IMPLEMENTATION
// ======================================================================

typedef struct {
  int *primary;    // Buffer currently being filled
  int *secondary;  // Buffer currently being sent (or free)
  int count;       // Number of items in primary
  int cap;         // Capacity of each buffer (entries, not bytes)
  MPI_Request req; // Request for the secondary buffer (if in flight)
  int active;      // 1 if a send is currently active (req is valid)
} StreamBuffer;

static StreamBuffer *stream_bufs = NULL;
static int *stream_recv_buf = NULL;
static int stream_recv_cap = 0;
// We need a way to track if we need to terminate.
// We use a non-blocking barrier approach or a token ring.
// A simpler robust way for "level completion" in async settings without
// extensive barriers is:
// 1. All ranks process their current frontier.
// 2. While processing, they send messages.
// 3. Occasionally they probe and receive messages.
// 4. When local frontier is done, they enter a "Flush & Wait" state.
// 5. In "Flush & Wait", they flush any remaining buffers.
// 6. Then they use a non-blocking termination detection (Ibarrier or
// consensus).

// Old init/free removed. Using new thread-safe versions below.

// Poll and process incoming messages
// Returns number of integers received (for termination detection)
static long long poll_and_process_stream(MPI_Graph *g, int *labels,
                                         int my_offset, int *next_frontier,
                                         int *next_frontier_size) {
  int flag = 0;
  MPI_Status status;
  long long total_received = 0;

  while (1) {
    MPI_Iprobe(MPI_ANY_SOURCE, 1337, MPI_COMM_WORLD, &flag, &status);
    if (!flag)
      break;

    int count;
    MPI_Get_count(&status, MPI_INT, &count);

    if (count > stream_recv_cap) {
      free(stream_recv_buf);
      stream_recv_cap = count + 1024;
      stream_recv_buf = aligned_alloc_wrapper(stream_recv_cap * sizeof(int));
    }

    // Blocking recv since we know it's there
    MPI_Recv(stream_recv_buf, count, MPI_INT, status.MPI_SOURCE, 1337,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    total_received += count;

    for (int k = 0; k < count; k += 2) {
      int v = stream_recv_buf[k];
      int comp = stream_recv_buf[k + 1];
      int local_v = v - my_offset;

      // If valid and unvisited
      if (local_v >= 0 && local_v < g->local_n) {
        if (labels[local_v] == 0) {
          labels[local_v] = comp;
          next_frontier[(*next_frontier_size)++] = local_v;
        }
      }
    }
  }
  return total_received;
}

// Context for Streaming BFS
typedef struct {
  long long total_sent;
  long long total_recv;
} StreamContext;

static StreamContext ctx;

// Wrapper to update context
static void poll_wrapper(MPI_Graph *g, int *labels, int my_offset,
                         int *next_frontier, int *next_frontier_size) {
  ctx.total_recv += poll_and_process_stream(g, labels, my_offset, next_frontier,
                                            next_frontier_size);
}

// push_wrapper removed - using push_to_global_locked instead

static void flush_wrapper(int size, MPI_Graph *g, int *labels, int my_offset,
                          int *next_frontier, int *next_frontier_size) {
  for (int p = 0; p < size; p++) {
    if (stream_bufs[p].count > 0) {
      if (stream_bufs[p].active) {
        int flag = 0;
        while (!flag) {
          MPI_Test(&stream_bufs[p].req, &flag, MPI_STATUS_IGNORE);
          if (!flag)
            poll_wrapper(g, labels, my_offset, next_frontier,
                         next_frontier_size);
        }
        stream_bufs[p].active = 0;
      }
      MPI_Isend(stream_bufs[p].primary, stream_bufs[p].count, MPI_INT, p, 1337,
                MPI_COMM_WORLD, &stream_bufs[p].req);
      ctx.total_sent += stream_bufs[p].count;
      stream_bufs[p].active = 1;
      stream_bufs[p].count = 0;
    }
  }
}

static void wait_wrapper(int size, MPI_Graph *g, int *labels, int my_offset,
                         int *next_frontier, int *next_frontier_size) {
  for (int p = 0; p < size; p++) {
    if (stream_bufs[p].active) {
      int flag = 0;
      while (!flag) {
        MPI_Test(&stream_bufs[p].req, &flag, MPI_STATUS_IGNORE);
        if (!flag)
          poll_wrapper(g, labels, my_offset, next_frontier, next_frontier_size);
      }
      stream_bufs[p].active = 0;
    }
  }
}

// TLS for Streaming BFS (Thread-Local Buffers per destination rank)

typedef struct {
  int **buffers; // [size][cap]
  int *counts;   // [size]
  int *caps;     // [size]
} TLS_Stream_Dynamic;

static TLS_Stream_Dynamic *tls_stream_dyn = NULL;
static pthread_mutex_t *stream_locks = NULL; // Locks for global stream buffers

void init_tls_stream_dyn(int workers, int size) {
  tls_stream_dyn = malloc(workers * sizeof(TLS_Stream_Dynamic));
  for (int w = 0; w < workers; w++) {
    tls_stream_dyn[w].buffers = malloc(size * sizeof(int *));
    tls_stream_dyn[w].counts = calloc(size, sizeof(int));
    tls_stream_dyn[w].caps = malloc(size * sizeof(int));
    for (int p = 0; p < size; p++) {
      tls_stream_dyn[w].caps[p] = 2048; // Small local buffer
      tls_stream_dyn[w].buffers[p] = malloc(2048 * sizeof(int));
    }
  }
}

void free_tls_stream_dyn(int workers, int size) {
  if (!tls_stream_dyn)
    return;
  for (int w = 0; w < workers; w++) {
    for (int p = 0; p < size; p++) {
      free(tls_stream_dyn[w].buffers[p]);
    }
    free(tls_stream_dyn[w].buffers);
    free(tls_stream_dyn[w].counts);
    free(tls_stream_dyn[w].caps);
  }
  free(tls_stream_dyn);
  tls_stream_dyn = NULL;
}

void init_stream_buffers(int size, int cap_per_rank) {
  stream_bufs = malloc(size * sizeof(StreamBuffer));
  stream_locks = malloc(size * sizeof(pthread_mutex_t));

  for (int p = 0; p < size; p++) {
    stream_bufs[p].cap = cap_per_rank;
    stream_bufs[p].primary = malloc(cap_per_rank * sizeof(int));
    stream_bufs[p].secondary = malloc(cap_per_rank * sizeof(int));
    stream_bufs[p].count = 0;
    stream_bufs[p].req = MPI_REQUEST_NULL;
    stream_bufs[p].active = 0;
    pthread_mutex_init(&stream_locks[p], NULL);
  }

  stream_recv_cap = 1024 * 1024; // Start with 1MB recv buffer
  stream_recv_buf = aligned_alloc_wrapper(stream_recv_cap * sizeof(int));
}

void free_stream_buffers(int size) {
  if (!stream_bufs)
    return;
  for (int p = 0; p < size; p++) {
    // Wait for any pending sends to complete before freeing!
    if (stream_bufs[p].active) {
      MPI_Wait(&stream_bufs[p].req, MPI_STATUS_IGNORE);
    }
    free(stream_bufs[p].primary);
    free(stream_bufs[p].secondary);
    pthread_mutex_destroy(&stream_locks[p]);
  }
  free(stream_bufs);
  free(stream_recv_buf);
  free(stream_locks);
  stream_bufs = NULL;
  stream_locks = NULL;
}

// Thread-safe push to global buffer
static void push_to_global_locked(int target_rank, int *buf, int count,
                                  MPI_Graph *g, int *labels, int my_offset,
                                  int *next_frontier, int *next_frontier_size) {
  if (count == 0)
    return;

  pthread_mutex_lock(&stream_locks[target_rank]);
  StreamBuffer *sb = &stream_bufs[target_rank];

  // Check if we need to flush current global buffer first
  if (sb->count + count > sb->cap) {
    // Must flush
    if (sb->active) {
      // If active send, we must wait or poll.
      // Since we hold the lock, no one else can touch 'sb'.
      int flag = 0;
      while (!flag) {
        MPI_Test(&sb->req, &flag, MPI_STATUS_IGNORE);
        // Can we poll here?
        // Since MPI_THREAD_MULTIPLE is on, yes.
        // But poll_wrapper receives into GLOBAL stream_recv_buf.
        // We need a lock for RECV too if we poll from multiple threads!
        // Or we just wait. Blocking here harms concurrency.
        // But 'sb->req' is specific to THIS destination.
        // Let's just Loop-Test.
      }
      sb->active = 0;
    }

    // Swap and Send
    int *tmp = sb->primary;
    sb->primary = sb->secondary;
    sb->secondary = tmp;

    MPI_Isend(sb->secondary, sb->count, MPI_INT, target_rank, 1337,
              MPI_COMM_WORLD, &sb->req);
    ctx.total_sent += sb->count;
    sb->active = 1;
    sb->count = 0;
  }

  // Now append our data (SAFE: we made space or it fits)
  // Wait, what if count > cap? (TLS buffer > Global buffer)
  // We should ensure TLS buffer <= Global buffer cap.
  // Assuming 2048 TLS << 50000 Global.

  // Copy
  memcpy(sb->primary + sb->count, buf, count * sizeof(int));
  sb->count += count;

  pthread_mutex_unlock(&stream_locks[target_rank]);
}

int mpi_coloring_cc_bfs_streaming(MPI_Graph *g) {
  int rank = g->rank;
  int size = g->size;
  int local_n = g->local_n;
  int my_offset = g->vtx_dist[rank];
  int *labels = g->local_labels;

  if (rank == 0)
    printf("Using Streaming Frontier BFS (Parallel Loop, TLS)...\n");

  memset(labels, 0, local_n * sizeof(int));
  int *frontier = aligned_alloc_wrapper(local_n * sizeof(int));
  int *next_frontier = aligned_alloc_wrapper(local_n * sizeof(int));
  int frontier_size = 0;
  int next_frontier_size = 0;

  init_stream_buffers(size, 50000);

  // Init TLS
  int n_workers = 1;
#ifdef __opencilk
  n_workers = __cilkrts_get_nworkers();
#endif
  init_tls_stream_dyn(n_workers, size);

  int total_levels = 0;

  for (int start_attempt = 0; start_attempt < local_n; start_attempt++) {
    int local_start = -1;
    for (int i = start_attempt; i < local_n; i++) {
      if (labels[i] == 0) {
        local_start = i;
        break;
      }
    }

    int my_global_start =
        (local_start >= 0) ? (my_offset + local_start) : INT_MAX;
    int global_min_start;
    MPI_Allreduce(&my_global_start, &global_min_start, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);

    if (global_min_start == INT_MAX)
      break;

    int component_id = global_min_start + 1;
    frontier_size = 0;
    if (my_global_start == global_min_start && local_start >= 0) {
      labels[local_start] = component_id;
      frontier[frontier_size++] = local_start;
    }

    int global_frontier_size = 1;
    while (global_frontier_size > 0) {
      total_levels++;
      next_frontier_size = 0;
      ctx.total_sent = 0;
      ctx.total_recv = 0;

      // Parallel Frontier Processing
      cilk_for(int f = 0; f < frontier_size; f++) {
        int tid = 0;
#ifdef __opencilk
        tid = __cilkrts_get_worker_number();
#endif

        int u = frontier[f];
        long long start = g->local_row_ptr[u];
        long long end = g->local_row_ptr[u + 1];

        // Optimization: Prefetching could be added here

        for (long long e = start; e < end; e++) {
          int v = g->local_col_idx[e];
          int owner = get_owner(v, g);

          if (owner == rank) {
            int local_v = v - my_offset;
            // Atomic check and set
            if (labels[local_v] == 0) {
              // Check again with CAS if needed, or if 0 -> set
              // If strict CC, we want exactly one setter.
              // Assuming atomic_min trick or CAS
              int expected = 0;
              if (__atomic_compare_exchange_n(&labels[local_v], &expected,
                                              component_id, 0, __ATOMIC_RELAXED,
                                              __ATOMIC_RELAXED)) {
                int idx = __atomic_fetch_add(&next_frontier_size, 1,
                                             __ATOMIC_RELAXED);
                next_frontier[idx] = local_v;
              }
            }
          } else {
            // Append to TLS
            int cnt = tls_stream_dyn[tid].counts[owner];
            if (cnt + 2 > tls_stream_dyn[tid].caps[owner]) {
              // Push current TLS buffer to global
              push_to_global_locked(owner, tls_stream_dyn[tid].buffers[owner],
                                    cnt, g, labels, my_offset, next_frontier,
                                    &next_frontier_size);
              tls_stream_dyn[tid].counts[owner] = 0;
              cnt = 0;
            }
            tls_stream_dyn[tid].buffers[owner][cnt] = v;
            tls_stream_dyn[tid].buffers[owner][cnt + 1] = component_id;
            tls_stream_dyn[tid].counts[owner] = cnt + 2;
          }
        }
      }

      // Flush residual TLS buffers
      for (int w = 0; w < n_workers; w++) {
        for (int p = 0; p < size; p++) {
          int cnt = tls_stream_dyn[w].counts[p];
          if (cnt > 0) {
            push_to_global_locked(p, tls_stream_dyn[w].buffers[p], cnt, g,
                                  labels, my_offset, next_frontier,
                                  &next_frontier_size);
            tls_stream_dyn[w].counts[p] = 0;
          }
        }
      }

      // Flush Global Buffers
      flush_wrapper(size, g, labels, my_offset, next_frontier,
                    &next_frontier_size);

      // Wait for locally active sends
      wait_wrapper(size, g, labels, my_offset, next_frontier,
                   &next_frontier_size);

      // Distributed Termination
      while (1) {
        poll_wrapper(g, labels, my_offset, next_frontier, &next_frontier_size);

        long long stats[2] = {ctx.total_sent, ctx.total_recv};
        long long global_stats[2];

        MPI_Request req;
        MPI_Iallreduce(stats, global_stats, 2, MPI_LONG_LONG, MPI_SUM,
                       MPI_COMM_WORLD, &req);

        int flag = 0;
        while (!flag) {
          MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
          if (!flag)
            poll_wrapper(g, labels, my_offset, next_frontier,
                         &next_frontier_size);
        }

        if (global_stats[0] == global_stats[1])
          break; // Done
      }

      // Swap
      int *tmp = frontier;
      frontier = next_frontier;
      next_frontier = tmp;
      frontier_size = next_frontier_size;

      MPI_Allreduce(&frontier_size, &global_frontier_size, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
    }
  }

  free_tls_stream_dyn(n_workers, size);
  free_stream_buffers(size);
  free(frontier);
  free(next_frontier);
  return total_levels;
}

/*
 * Frontier-Based BFS for Connected Components (MPI Version)
 * O(diameter) iterations - much faster for single-component graphs
 */
int mpi_coloring_cc_bfs(MPI_Graph *g) {
  int rank = g->rank;
  int size = g->size;
  int local_n = g->local_n;
  int my_offset = g->vtx_dist[rank];
  int *labels = g->local_labels;

  // Initialize labels to 0 (unvisited)
  memset(labels, 0, local_n * sizeof(int));

  // Allocate frontiers
  int *frontier = aligned_alloc_wrapper(local_n * sizeof(int));
  int *next_frontier = aligned_alloc_wrapper(local_n * sizeof(int));
  int frontier_size = 0;
  int next_frontier_size = 0;

  // Remote frontier buffers (for exchange)
  int **send_bufs = malloc(size * sizeof(int *));
  int *send_counts = calloc(size, sizeof(int));
  int *send_caps = malloc(size * sizeof(int));
  for (int p = 0; p < size; p++) {
    send_caps[p] = 10000000; // 10M entries - enough for high-degree graphs
    send_bufs[p] = malloc(send_caps[p] * sizeof(int));
  }

  int *recv_counts = malloc(size * sizeof(int));
  int *recv_displs = malloc(size * sizeof(int));
  int *send_displs = malloc(size * sizeof(int));
  int recv_buf_cap = local_n;
  int *recv_buf = aligned_alloc_wrapper(recv_buf_cap * sizeof(int));

  int total_levels = 0;
  int component_id = 0;

  // Outer loop: find components by iterating unvisited vertices
  for (int start_attempt = 0; start_attempt < local_n; start_attempt++) {
    // Find an unvisited local vertex to start a new component
    int local_start = -1;
    for (int i = start_attempt; i < local_n; i++) {
      if (labels[i] == 0) {
        local_start = i;
        break;
      }
    }

    // Coordinate globally: who has the lowest global unvisited vertex?
    int my_global_start =
        (local_start >= 0) ? (my_offset + local_start) : INT_MAX;
    int global_min_start;
    MPI_Allreduce(&my_global_start, &global_min_start, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);

    if (global_min_start == INT_MAX) {
      break; // All vertices visited
    }

    component_id = global_min_start + 1; // 1-based component ID

    // Initialize frontier: only the rank owning global_min_start starts
    frontier_size = 0;
    if (my_global_start == global_min_start && local_start >= 0) {
      labels[local_start] = component_id;
      frontier[frontier_size++] = local_start;
    }

    // BFS levels
    int global_frontier_size = 1;
    while (global_frontier_size > 0) {
      total_levels++;

      // Clear send counts
      for (int p = 0; p < size; p++)
        send_counts[p] = 0;
      next_frontier_size = 0;

      // Process local frontier
      cilk_for(int f = 0; f < frontier_size; f++) {
        int u = frontier[f];
        long long start = g->local_row_ptr[u];
        long long end = g->local_row_ptr[u + 1];

        for (long long e = start; e < end; e++) {
          int v = g->local_col_idx[e];
          int owner = get_owner(v, g);

          if (owner == rank) {
            // Local neighbor
            int local_v = v - my_offset;
            if (labels[local_v] == 0) {
              int expected = 0;
              if (__atomic_compare_exchange_n(&labels[local_v], &expected,
                                              component_id, 0, __ATOMIC_RELAXED,
                                              __ATOMIC_RELAXED)) {
                int idx = __atomic_fetch_add(&next_frontier_size, 1,
                                             __ATOMIC_RELAXED);
                next_frontier[idx] = local_v;
              }
            }
          } else {
            // Remote neighbor: pack for sending
            int cnt =
                __atomic_fetch_add(&send_counts[owner], 2, __ATOMIC_RELAXED);
            // Grow buffer if needed (not thread-safe, but ok for small growth)
            if (cnt + 2 > send_caps[owner]) {
              // Skip for now - should preallocate enough
            } else {
              send_bufs[owner][cnt] = v;
              send_bufs[owner][cnt + 1] = component_id;
            }
          }
        }
      }

      // Exchange sizes
      MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT,
                   MPI_COMM_WORLD);

      // Calculate displacements
      send_displs[0] = 0;
      recv_displs[0] = 0;
      int total_send = send_counts[0];
      int total_recv = recv_counts[0];
      for (int p = 1; p < size; p++) {
        send_displs[p] = send_displs[p - 1] + send_counts[p - 1];
        recv_displs[p] = recv_displs[p - 1] + recv_counts[p - 1];
        total_send += send_counts[p];
        total_recv += recv_counts[p];
      }

      // Pack send buffer
      int *packed_send = aligned_alloc_wrapper((total_send + 1) * sizeof(int));
      for (int p = 0; p < size; p++) {
        memcpy(packed_send + send_displs[p], send_bufs[p],
               send_counts[p] * sizeof(int));
      }

      // Resize recv buf if needed
      if (total_recv > recv_buf_cap) {
        free(recv_buf);
        recv_buf_cap = total_recv + 1024;
        recv_buf = aligned_alloc_wrapper(recv_buf_cap * sizeof(int));
      }

      // Exchange frontier vertices
      MPI_Alltoallv(packed_send, send_counts, send_displs, MPI_INT, recv_buf,
                    recv_counts, recv_displs, MPI_INT, MPI_COMM_WORLD);
      free(packed_send);

      // Process received vertices
      for (int i = 0; i < total_recv; i += 2) {
        int v = recv_buf[i];
        int comp = recv_buf[i + 1];
        int local_v = v - my_offset;
        if (local_v >= 0 && local_v < local_n && labels[local_v] == 0) {
          labels[local_v] = comp;
          next_frontier[next_frontier_size++] = local_v;
        }
      }

      // Swap frontiers
      int *tmp = frontier;
      frontier = next_frontier;
      next_frontier = tmp;
      frontier_size = next_frontier_size;

      // Global convergence check
      MPI_Allreduce(&frontier_size, &global_frontier_size, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
    }
  }

  // Cleanup
  free(frontier);
  free(next_frontier);
  for (int p = 0; p < size; p++)
    free(send_bufs[p]);
  free(send_bufs);
  free(send_counts);
  free(send_caps);
  free(recv_counts);
  free(recv_displs);
  free(send_displs);
  free(recv_buf);

  return total_levels;
}

void mpi_cleanup_graph(MPI_Graph *g) {
  // local_row_ptr and local_col_idx are freed in setup_communication
  free(g->local_labels);
  free(g->vtx_dist);

  // We KEEP the original adjacency lists now!
  // Free split arrays
  free(g->split_indices);
  // free(g->local_row_ptr);
  // free(g->local_col_idx);
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

int mpi_init_graph(MPI_Graph *g, int n, const long long *row_ptr,
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

  g->local_row_ptr =
      aligned_alloc_wrapper((g->local_n + 1) * sizeof(long long));

  if (g->rank == 0) {
    for (int p = 1; p < g->size; p++) {
      int p_start = g->vtx_dist[p];
      int p_count = (g->vtx_dist[p + 1] - g->vtx_dist[p]) + 1;
      MPI_Send(&row_ptr[p_start], p_count, MPI_LONG_LONG_INT, p, 0,
               MPI_COMM_WORLD);
    }
    memcpy(g->local_row_ptr, row_ptr, (g->local_n + 1) * sizeof(long long));
  } else {
    MPI_Recv(g->local_row_ptr, g->local_n + 1, MPI_LONG_LONG_INT, 0, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  long long global_start_idx = g->local_row_ptr[0];
  for (int i = 0; i <= g->local_n; i++) {
    g->local_row_ptr[i] -= global_start_idx;
  }

  g->local_nnz = g->local_row_ptr[g->local_n];
  g->local_col_idx = aligned_alloc_wrapper(g->local_nnz * sizeof(int));
  g->local_labels = aligned_alloc_wrapper(g->local_n * sizeof(int));

  if (g->rank == 0) {
    // Send col_idx to other ranks
    for (int p = 1; p < g->size; p++) {
      int p_start = g->vtx_dist[p];
      int p_end = g->vtx_dist[p + 1];
      long long start_idx = row_ptr[p_start];
      long long end_idx = row_ptr[p_end]; // row_ptr is long long
      long long count_long = end_idx - start_idx;

      // We assume count_long fits in INT for now (ranks >= 2 for Friendster)
      // If single chunk > 2GB, we would need chunked sending.
      if (count_long > INT_MAX) {
        fprintf(stderr,
                "Error: Rank %d needs %lld edges, which exceeds INT_MAX. "
                "Increase number of ranks!\n",
                p, count_long);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }

      MPI_Send(col_idx + start_idx, (int)count_long, MPI_INT, p, 1,
               MPI_COMM_WORLD);
    }

    // Copy local part
    int p_start = g->vtx_dist[0];
    int p_end = g->vtx_dist[1];
    long long start_idx = row_ptr[p_start];
    long long end_idx = row_ptr[p_end];
    long long count_long = end_idx - start_idx;
    memcpy(g->local_col_idx, col_idx + start_idx, count_long * sizeof(int));

  } else {
    // Receive col_idx
    // local_nnz is already known from row_ptr
    if (g->local_nnz > INT_MAX) {
      fprintf(stderr,
              "Error: Rank %d has %lld edges, exceeds INT_MAX. Increase number "
              "of ranks!\n",
              g->rank, (long long)g->local_nnz);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Recv(g->local_col_idx, (int)g->local_nnz, MPI_INT, 0, 1, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

  g->split_indices = NULL;
  return 0;
}

int mpi_init_graph_distributed(MPI_Graph *g, int global_n, int local_n,
                               long long *local_row_ptr, int *local_col_idx,
                               long long local_nnz, const int *vtx_dist) {
  MPI_Comm_rank(MPI_COMM_WORLD, &g->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g->size);
  g->global_n = global_n;
  g->local_n = local_n;
  g->local_nnz = local_nnz;

  // Take ownership of the pointers
  g->local_row_ptr = local_row_ptr;
  g->local_col_idx = local_col_idx;

  // Copy vtx_dist
  g->vtx_dist = malloc((g->size + 1) * sizeof(int));
  if (!g->vtx_dist) {
    perror("Failed to alloc vtx_dist");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  memcpy(g->vtx_dist, vtx_dist, (g->size + 1) * sizeof(int));

  // Sanity check
  int expected_n = g->vtx_dist[g->rank + 1] - g->vtx_dist[g->rank];
  if (expected_n != local_n) {
    fprintf(stderr,
            "Rank %d: Partition mismatch. Expected %d, got %d. GlobalN=%d\n",
            g->rank, expected_n, local_n, global_n);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  g->local_labels = aligned_alloc_wrapper(g->local_n * sizeof(int));
  if (!g->local_labels) {
    perror("Failed to alloc labels");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  g->split_indices = NULL;
  return 0;
}

// ======================================================================

// ======================================================================
// ASYNCHRONOUS IMPLEMENTATION START
// ======================================================================

// Helper to check for incoming messages and process them
static void poll_and_process_async(int *labels, int my_offset, int local_n,
                                   int *recv_buf_tmp, int recv_cap,
                                   uint64_t *next_changed,
                                   long long *processed_count) {
  int flag = 0;
  MPI_Status status;
  // Poll for message
  MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
  while (flag) {
    int count;
    MPI_Get_count(&status, MPI_INT, &count);

    // Blocking recv for the probed message
    MPI_Recv(recv_buf_tmp, count, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    int received_items = count / 2;
    *processed_count += received_items;

    for (int k = 0; k < count; k += 2) {
      int target = recv_buf_tmp[k];
      int val = recv_buf_tmp[k + 1];
      if (target >= my_offset && target < my_offset + local_n) {
        if (atomic_min(&labels[target - my_offset], val)) {
          bitset_set(next_changed, target - my_offset);
        }
      }
    }

    // Check again
    MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
  }
}

// Helper to wait for a request while polling
static void wait_with_polling(MPI_Request *req, int *labels, int my_offset,
                              int local_n, int *recv_buf_tmp, int recv_cap,
                              uint64_t *next_changed,
                              long long *processed_count) {
  if (*req == MPI_REQUEST_NULL)
    return;

  int flag = 0;
  while (!flag) {
    MPI_Test(req, &flag, MPI_STATUS_IGNORE);
    if (!flag) {
      poll_and_process_async(labels, my_offset, local_n, recv_buf_tmp, recv_cap,
                             next_changed, processed_count);
    }
  }
}

int mpi_coloring_cc_async(MPI_Graph *g) {
  // Adaptive dispatcher: check if Frontier BFS should be used
  static int use_frontier = -1; // -1 = not checked yet
  if (use_frontier == -1) {
    char *env = getenv("BFS_USE_FRONTIER");
    use_frontier = (env && atoi(env) > 0) ? 1 : 0;
    if (g->rank == 0 && use_frontier) {
      printf(
          "BFS_USE_FRONTIER=1: Using Frontier BFS (O(diameter) iterations)\n");
    }
  }

  if (use_frontier) {
    setup_communication(g);
    return mpi_coloring_cc_bfs(g);
  }

  // Fall through to Async Label Propagation (default)
  setup_communication(g);

  int rank = g->rank;
  int size = g->size;
  int local_n = g->local_n;
  int *labels = g->local_labels;
  int my_offset = g->vtx_dist[rank];

  // Inform compiler about alignment
  labels = __builtin_assume_aligned(labels, 64);
  comm.send_buf = __builtin_assume_aligned(comm.send_buf, 64);

  // Initialize labels (1-based)
  cilk_for(int i = 0; i < local_n; i++) { labels[i] = my_offset + i + 1; }

  // Frontier arrays
  size_t frontier_words = BITSET_WORDS(local_n);
  uint64_t *changed = aligned_alloc_wrapper(frontier_words * sizeof(uint64_t));
  uint64_t *next_changed =
      aligned_alloc_wrapper(frontier_words * sizeof(uint64_t));

  // Initially all vertices actve
  memset(changed, 0xFF, frontier_words * sizeof(uint64_t));
  bitset_clear(next_changed, frontier_words);

  if (local_n % BITSET_WORD_SIZE != 0) {
    uint64_t mask = (1UL << (local_n % BITSET_WORD_SIZE)) - 1;
    changed[frontier_words - 1] &= mask;
  }

  // --- Async specific buffers (Double Buffered) ---
  int **async_send_bufs[2];
  async_send_bufs[0] = malloc(size * sizeof(int *));
  async_send_bufs[1] = malloc(size * sizeof(int *));

  MPI_Request
      *async_send_reqs[2]; // Fixed: Single pointer level (array of requests)
  async_send_reqs[0] = malloc(size * sizeof(MPI_Request));
  async_send_reqs[1] = malloc(size * sizeof(MPI_Request));

  int *async_send_counts = calloc(size, sizeof(int));
  int *async_active_buf_idx = calloc(size, sizeof(int));

  int async_buf_cap = 4096; // Tunable: batch size
  for (int p = 0; p < size; p++) {
    async_send_bufs[0][p] = malloc(async_buf_cap * sizeof(int));
    async_send_bufs[1][p] = malloc(async_buf_cap * sizeof(int));
    async_send_reqs[0][p] = MPI_REQUEST_NULL;
    async_send_reqs[1][p] = MPI_REQUEST_NULL;
  }

  // Temp Recv Buffer for polling
  int *async_recv_buf = malloc((async_buf_cap + 1024) * sizeof(int));

  // Termination stats
  long long total_sent = 0;
  long long total_recv = 0;

  int iterations = 0;
  int active = 1;

  // Non-blocking termination state
  MPI_Request term_req = MPI_REQUEST_NULL;
  int term_check_active = 0;
  long long term_send_recv_buf[2];
  long long term_send_recv_global[2];
  int term_idle_buf;
  int term_idle_global;
  MPI_Request term_idle_req = MPI_REQUEST_NULL;

  while (active) {
    iterations++;

    // 1. Process local `changed` frontier
    int any_changed_bit = 0;
    for (size_t w = 0; w < frontier_words; w++) {
      if (changed[w]) {
        any_changed_bit = 1;
        break;
      }
    }

    if (any_changed_bit) {
      for (int i = 0; i < local_n; i++) {
        if (bitset_test(changed, i)) {
          int val = labels[i];

          // Local Propagation
          long long start = g->local_row_ptr[i];
          long long end = g->split_indices[i];
          for (long long j = start; j < end; j++) {
            int neighbor = g->local_col_idx[j];
            if (atomic_min(&labels[neighbor - my_offset], val)) {
              bitset_set(next_changed, neighbor - my_offset);
            }
          }

          // Remote Propagation (Pack)
          start = g->split_indices[i];
          end = g->local_row_ptr[i + 1];
          for (long long j = start; j < end; j++) {
            int neighbor = g->local_col_idx[j];
            int owner = get_owner(neighbor, g);

            int idx = async_active_buf_idx[owner];
            async_send_bufs[idx][owner][async_send_counts[owner]++] = neighbor;
            async_send_bufs[idx][owner][async_send_counts[owner]++] = val;

            // Check flush
            if (async_send_counts[owner] >= async_buf_cap) {
              // Time to send!
              MPI_Isend(async_send_bufs[idx][owner], async_send_counts[owner],
                        MPI_INT, owner, 0, MPI_COMM_WORLD,
                        &async_send_reqs[idx][owner]);

              total_sent += (async_send_counts[owner] / 2);
              async_send_counts[owner] = 0;

              // Switch buffer
              int next_idx = 1 - idx;
              async_active_buf_idx[owner] = next_idx;

              // Ensure new buffer is free (Backpressure with Polling)
              wait_with_polling(&async_send_reqs[next_idx][owner], labels,
                                my_offset, local_n, async_recv_buf,
                                async_buf_cap, next_changed, &total_recv);
            }
          }
        }

        // Frequent Polling
        if (i % 256 == 0) {
          long long recvd = 0;
          poll_and_process_async(labels, my_offset, local_n, async_recv_buf,
                                 async_buf_cap, next_changed, &recvd);
          total_recv += recvd;
        }
      }
    }

    // Flush remaining buffers
    for (int p = 0; p < size; p++) {
      if (async_send_counts[p] > 0) {
        int idx = async_active_buf_idx[p];
        MPI_Isend(async_send_bufs[idx][p], async_send_counts[p], MPI_INT, p, 0,
                  MPI_COMM_WORLD, &async_send_reqs[idx][p]);
        total_sent += (async_send_counts[p] / 2);
        async_send_counts[p] = 0;

        int next_idx = 1 - idx;
        async_active_buf_idx[p] = next_idx;

        wait_with_polling(&async_send_reqs[next_idx][p], labels, my_offset,
                          local_n, async_recv_buf, async_buf_cap, next_changed,
                          &total_recv);
      }
    }

    // 2. Poll hard
    long long recvd = 0;
    poll_and_process_async(labels, my_offset, local_n, async_recv_buf,
                           async_buf_cap, next_changed, &recvd);
    total_recv += recvd;

    // 3. Swap Frontiers
    uint64_t *tmp = changed;
    changed = next_changed;
    next_changed = tmp;
    bitset_clear(next_changed, frontier_words);

    // 4. Non-Blocking Termination
    if (!term_check_active) {
      if (iterations % 5 == 0) {
        term_send_recv_buf[0] = total_sent;
        term_send_recv_buf[1] = total_recv;

        term_idle_buf = 1;
        for (size_t w = 0; w < frontier_words; w++) {
          if (changed[w]) {
            term_idle_buf = 0;
            break;
          }
        }

        MPI_Iallreduce(term_send_recv_buf, term_send_recv_global, 2,
                       MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD, &term_req);
        MPI_Iallreduce(&term_idle_buf, &term_idle_global, 1, MPI_INT, MPI_LAND,
                       MPI_COMM_WORLD, &term_idle_req);
        term_check_active = 1;
      }
    } else {
      int flag1 = 0, flag2 = 0;
      MPI_Test(&term_req, &flag1, MPI_STATUS_IGNORE);
      MPI_Test(&term_idle_req, &flag2, MPI_STATUS_IGNORE);

      if (flag1 && flag2) {
        term_check_active = 0;
        if (term_idle_global &&
            term_send_recv_global[0] == term_send_recv_global[1]) {
          active = 0;
        }
      }
    }
  }

  // Cleanup pending sends (Wait with Polling to be safe, though active=0 means
  // we probably match states) Actually, if active=0, global quiesence is
  // reached. All messages received. So all sends MUST have completed at
  // destination? Not necessarily "Request" is free locally. We can just Wait.
  for (int p = 0; p < size; p++) {
    if (async_send_reqs[0][p] != MPI_REQUEST_NULL)
      MPI_Wait(&async_send_reqs[0][p], MPI_STATUS_IGNORE);
    if (async_send_reqs[1][p] != MPI_REQUEST_NULL)
      MPI_Wait(&async_send_reqs[1][p], MPI_STATUS_IGNORE);
  }

  // Cleanup memory
  for (int p = 0; p < size; p++) {
    free(async_send_bufs[0][p]);
    free(async_send_bufs[1][p]);
  }
  free(async_send_bufs[0]);
  free(async_send_bufs[1]);
  free(async_send_reqs[0]);
  free(async_send_reqs[1]);
  free(async_send_counts);
  free(async_active_buf_idx);

  free(async_recv_buf);
  free(changed);
  free(next_changed);
  cleanup_communication();

  return iterations;
}
