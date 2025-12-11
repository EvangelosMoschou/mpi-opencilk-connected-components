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
        if (neighbor < 0) { // Should not happen with unsigned/int
          // Handle error or ignore
          owner = 0;
        } else {
          owner = get_owner(neighbor, g);
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
      int deg = (g->internal_row_ptr[i + 1] - g->internal_row_ptr[i]) +
                (g->remote_row_ptr[i + 1] - g->remote_row_ptr[i]);
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
        int local_avg_deg =
            (local_n > 0) ? (g->internal_row_ptr[local_n] / local_n) : 0;

        // Internal neighbors
        int start = g->internal_row_ptr[local_max_v];
        int end = g->internal_row_ptr[local_max_v + 1];
        for (int j = start; j < end; j++) {
          int neighbor = g->internal_col_idx[j];
          if (atomic_min(&labels[neighbor - my_offset], val)) {
            bitset_set(next_changed, neighbor - my_offset);

            // Hub Push: If neighbor is a hub, push to its neighbors
            // Note: neighbor is local (internal edge)
            int neighbor_local_idx = neighbor - my_offset;
            int neighbor_deg = (g->internal_row_ptr[neighbor_local_idx + 1] -
                                g->internal_row_ptr[neighbor_local_idx]);
            // We ignore remote degree for simplicity in this heuristic check

            if (neighbor_deg > local_avg_deg) {
              int n_start = g->internal_row_ptr[neighbor_local_idx];
              int n_end = g->internal_row_ptr[neighbor_local_idx + 1];
              for (int k = n_start; k < n_end; k++) {
                int w = g->internal_col_idx[k];
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

        // Optimization: Early Retirement
        // If I already have the global minimum label (1), I don't need to do
        // anything unless I need to propagate to neighbors who might not have
        // it. But if I am 1, I only propagate 1. Wait, if I am 1, I should
        // propagate to neighbors. But if I was *already* 1 in previous round
        // and processed, I wouldn't be in 'changed'. If I just became 1, I need
        // to propagate. So this check might be redundant if 'changed' logic is
        // correct. However, if we scan all vertices (not using frontier), it's
        // useful. Here we use frontier 'changed'. But let's keep it for
        // safety/clarity or if we switch to dense scan. Actually, for
        // frontier-based, if we are in 'changed', we MUST propagate. So "Early
        // Retirement" in the sense of "skip processing" applies if we are NOT
        // in changed. But maybe the user meant "if I see a neighbor with 1, I
        // take it and stop looking at other neighbors". That is handled by
        // atomic_min logic.

        // Let's stick to the request: "if (my_label == 1) continue;" at top of
        // loop. But if my_label is 1, I *should* propagate 1 to neighbors! If I
        // continue, I stop propagating. That would be bad. UNLESS the logic is:
        // "If I have 1, I have converged. My neighbors will get it from me."
        // No, I am the active vertex. I need to push to neighbors.

        // Re-reading user request: "If a vertex acquires the 0 (global minimum)
        // label, it permanently retires." This implies it doesn't need to be
        // processed again. But it needs to process ONCE to tell neighbors.
        // After that, it is removed from 'changed'.
        // So the frontier logic naturally handles retirement.
        // Maybe the optimization is for the *pull* model? Or dense scan?
        // In our push-based frontier model, if I am 1, I propagate. Next round
        // I am not in 'changed'. So I am retired.

        // However, maybe the user means: "If I am iterating neighbors and I
        // find a neighbor is already 1, skip it?" Or "If I am 1, I don't need
        // to check my parent pointer?"

        // Let's implement the check as requested but ensure we don't break
        // propagation. If I am 1, I still need to propagate. But maybe I don't
        // need path compression?
        if (val == 1) {
          // I am 1. I don't need path compression.
          // I just propagate 1 to neighbors.
        } else {
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

        int start = g->internal_row_ptr[i];
        int end = g->internal_row_ptr[i + 1];

        int limit = end;
        if (iterations <= sampling_rounds) {
          if (start + sampling_k < end)
            limit = start + sampling_k;
        }
        for (int j = start; j < limit; j++) {
          // Optimization: Software Prefetching
          if (prefetch_dist > 0) {
            // Stage 1: Prefetch neighbor index
            if (j + prefetch_dist * 2 < limit) {
              __builtin_prefetch(&g->internal_col_idx[j + prefetch_dist * 2], 0,
                                 3);
            }
            // Stage 2: Prefetch label
            if (j + prefetch_dist < limit) {
              int neighbor_next = g->internal_col_idx[j + prefetch_dist];
              __builtin_prefetch(&labels[neighbor_next - my_offset], 0, 3);
            }
          }

          int neighbor = g->internal_col_idx[j];
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

          int start = g->remote_row_ptr[u];
          int end = g->remote_row_ptr[u + 1];

          int limit = end;
          if (iterations <= sampling_rounds) {
            if (start + sampling_k < end)
              limit = start + sampling_k;
          }
          for (int j = start; j < limit; j++) {
            // Prefetching
            if (prefetch_dist > 0 && j + prefetch_dist < limit) {
              __builtin_prefetch(&g->remote_col_idx[j + prefetch_dist], 0, 3);
              __builtin_prefetch(&g->remote_owners[j + prefetch_dist], 0, 3);
            }

            int neighbor = g->remote_col_idx[j];
            int owner = g->remote_owners[j];

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

          int start = g->remote_row_ptr[u];
          int end = g->remote_row_ptr[u + 1];

          int limit = end;
          if (iterations <= sampling_rounds) {
            if (start + sampling_k < end)
              limit = start + sampling_k;
          }
          for (int j = start; j < limit; j++) {
            if (prefetch_dist > 0 && j + prefetch_dist < limit) {
              __builtin_prefetch(&g->remote_col_idx[j + prefetch_dist], 0, 3);
              __builtin_prefetch(&g->remote_owners[j + prefetch_dist], 0, 3);
            }

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

int mpi_init_graph_distributed(MPI_Graph *g, int global_n, int local_n,
                               int *local_row_ptr, int *local_col_idx,
                               int local_nnz, const int *vtx_dist) {
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
  return 0;
}
