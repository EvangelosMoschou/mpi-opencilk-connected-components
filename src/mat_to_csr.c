#define _POSIX_C_SOURCE 200809L
#include <fcntl.h>
#include <inttypes.h>
#include <matio.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __opencilk
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#else
#define cilk_for for
#define cilk_spawn
#define cilk_sync
#endif

#include "mat_to_csr.h"

// Helper for aligned allocation (64-byte alignment for AVX-512/Cache lines)
void *aligned_alloc_wrapper(size_t size) {
  void *ptr = NULL;
  if (posix_memalign(&ptr, 64, size) != 0) {
    return NULL;
  }
  return ptr;
}

void clean_csr_vectors(CSR_info vecs) {
  free(vecs.row_ptr);
  free(vecs.col_idx);
}

// Compare function for qsort
static int cmp_int(const void *a, const void *b) {
  return (*(int *)a - *(int *)b);
}

// Insertion sort for small arrays (faster than qsort)
static void insertion_sort(int *arr, int n) {
  for (int i = 1; i < n; i++) {
    int key = arr[i];
    int j = i - 1;
    while (j >= 0 && arr[j] > key) {
      arr[j + 1] = arr[j];
      j--;
    }
    arr[j + 1] = key;
  }
}

// Sort array and remove duplicates in-place. Returns new count.
static int sort_deduplicate_row(int *arr, int count) {
  if (count <= 0)
    return 0;
  if (count < 32) {
    insertion_sort(arr, count);
  } else {
    qsort(arr, count, sizeof(int), cmp_int);
  }
  int unique = 0;
  for (int i = 0; i < count; i++) {
    if (i == 0 || arr[i] != arr[i - 1]) {
      arr[unique++] = arr[i];
    }
  }
  return unique;
}

typedef struct {
  int id;
  int degree;
} NodeDegree;

static int cmp_degree_desc(const void *a, const void *b) {
  NodeDegree *n1 = (NodeDegree *)a;
  NodeDegree *n2 = (NodeDegree *)b;
  if (n1->degree != n2->degree) {
    return n2->degree - n1->degree; // Descending
  }
  return n1->id - n2->id; // Ascending ID as tie-breaker
}

// Check if graph is symmetric
static int is_symmetric(const CSR_info *csr) {
  int symmetric = 1;
  // We can check a subset of edges or full check. Full check is safer.
  // Since we sort rows, we can check if u->v implies v->u exists.
  // This is expensive (O(E log D)), so we might just assume NO and symmetrize

  // if unsure, but the user asked for a check. Optimization: Check random
  // samples? Or just do full check in parallel? Let's do a full check but
  // parallelized.

  // Note: This check assumes rows are sorted!
  cilk_for(int i = 0; i < csr->nrows; i++) {
    if (!symmetric)
      continue; // Early exit (best effort in parallel)
    for (long long j = csr->row_ptr[i]; j < csr->row_ptr[i + 1]; j++) {
      int neighbor = csr->col_idx[j];
      if (neighbor == i)
        continue; // Self-loop is symmetric

      // Check if edge neighbor->i exists
      int found = 0;
      long long start = csr->row_ptr[neighbor];
      long long end = csr->row_ptr[neighbor + 1];

      // Binary search since rows are sorted
      long long low = start, high = end - 1;
      while (low <= high) {
        long long mid = low + (high - low) / 2;
        if (csr->col_idx[mid] == i) {
          found = 1;
          break;
        } else if (csr->col_idx[mid] < i) {
          low = mid + 1;
        } else {
          high = mid - 1;
        }
      }

      if (!found) {
        symmetric = 0;
        // No break here to avoid race on symmetric flag (it's just 0 or 1)
      }
    }
  }
  return symmetric;
}

// Symmetrize graph: A = A U A^T
static void symmetrize_graph(CSR_info *csr) {
  int n = csr->nrows;
  long long *row_ptr = csr->row_ptr;
  int *col_idx = csr->col_idx;

  // 1. Count degrees for transpose
  int *trans_deg = calloc(n, sizeof(int));
  for (long long i = 0; i < csr->nzcount; i++) {
    trans_deg[col_idx[i]]++;
  }

  // 2. Estimate new degrees (upper bound: orig + trans)
  // We will filter duplicates later
  long long *new_row_ptr = aligned_alloc_wrapper((n + 1) * sizeof(long long));
  new_row_ptr[0] = 0;
  for (int i = 0; i < n; i++) {
    int orig_deg = row_ptr[i + 1] - row_ptr[i];
    new_row_ptr[i + 1] = new_row_ptr[i] + orig_deg + trans_deg[i];
  }

  long long max_nz = new_row_ptr[n];
  int *temp_col_idx = aligned_alloc_wrapper(max_nz * sizeof(int));

  // 3. Fill with both A and A^T
  long long *offsets = calloc(n, sizeof(long long));

  // Copy original A
  cilk_for(int r = 0; r < n; r++) {
    long long start = new_row_ptr[r];
    for (long long j = row_ptr[r]; j < row_ptr[r + 1]; j++) {
      int c = col_idx[j];
      long long dest =
          start + __atomic_fetch_add(&offsets[r], 1, __ATOMIC_RELAXED);
      temp_col_idx[dest] = c;

      // Add transpose A^T (c -> r)
      long long dest_t =
          new_row_ptr[c] + __atomic_fetch_add(&offsets[c], 1, __ATOMIC_RELAXED);
      temp_col_idx[dest_t] = r;
    }
  }

  free(trans_deg);
  free(offsets);

  // 4. Sort and remove duplicates
  // We need to compact the array
  long long *final_row_ptr = aligned_alloc_wrapper((n + 1) * sizeof(long long));
  final_row_ptr[0] = 0;

  int *new_counts = calloc(n, sizeof(int));

  cilk_for(int i = 0; i < n; i++) {
    long long start = new_row_ptr[i];
    int count = new_row_ptr[i + 1] - start;
    new_counts[i] = sort_deduplicate_row(&temp_col_idx[start], count);
  }

  // Prefix sum for final row pointers
  for (int i = 0; i < n; i++) {
    final_row_ptr[i + 1] = final_row_ptr[i] + new_counts[i];
  }

  long long final_nz = final_row_ptr[n];
  int *final_col_idx = aligned_alloc_wrapper(final_nz * sizeof(int));

  cilk_for(int i = 0; i < n; i++) {
    long long src_start = new_row_ptr[i];
    long long dest_start = final_row_ptr[i];
    int count = new_counts[i];
    memcpy(&final_col_idx[dest_start], &temp_col_idx[src_start],
           count * sizeof(int));
  }

  free(new_row_ptr);
  free(temp_col_idx);
  free(new_counts);
  free(csr->row_ptr);
  free(csr->col_idx);

  csr->row_ptr = final_row_ptr;
  csr->col_idx = final_col_idx;
  csr->nzcount = final_nz;
}

// Reorder CSR graph by descending node degree (in-place)
static void reorder_csr_by_degree(CSR_info *csr) {
  int n = csr->nrows;

  // 1. Compute permutation from degree
  NodeDegree *nodes = malloc(n * sizeof(NodeDegree));
  if (!nodes) {
    printf("Reorder: alloc failed\n");
    return;
  }

  cilk_for(int i = 0; i < n; i++) {
    nodes[i].id = i;
    nodes[i].degree = csr->row_ptr[i + 1] - csr->row_ptr[i];
  }
  qsort(nodes, n, sizeof(NodeDegree), cmp_degree_desc);

  int *old_to_new = malloc(n * sizeof(int));
  int *new_to_old = malloc(n * sizeof(int));
  if (!old_to_new || !new_to_old) {
    free(nodes);
    free(old_to_new);
    free(new_to_old);
    printf("Reorder: alloc failed\n");
    return;
  }

  cilk_for(int i = 0; i < n; i++) {
    new_to_old[i] = nodes[i].id;
    old_to_new[nodes[i].id] = i;
  }
  free(nodes);

  // 2. Build new CSR
  long long *new_row_ptr = aligned_alloc_wrapper((n + 1) * sizeof(long long));
  int *new_col_idx = aligned_alloc_wrapper(csr->nzcount * sizeof(int));
  if (!new_row_ptr || !new_col_idx) {
    free(old_to_new);
    free(new_to_old);
    if (new_row_ptr)
      free(new_row_ptr);
    if (new_col_idx)
      free(new_col_idx);
    printf("Reorder: alloc failed\n");
    return;
  }

  new_row_ptr[0] = 0;
  for (int i = 0; i < n; i++) {
    int old_u = new_to_old[i];
    new_row_ptr[i + 1] =
        new_row_ptr[i] + (csr->row_ptr[old_u + 1] - csr->row_ptr[old_u]);
  }

  cilk_for(int i = 0; i < n; i++) {
    int old_u = new_to_old[i];
    long long src = csr->row_ptr[old_u];
    long long dst = new_row_ptr[i];
    int cnt = csr->row_ptr[old_u + 1] - src;
    for (int j = 0; j < cnt; j++) {
      new_col_idx[dst + j] = old_to_new[csr->col_idx[src + j]];
    }
    sort_deduplicate_row(&new_col_idx[dst], cnt);
  }

  free(old_to_new);
  free(new_to_old);
  free(csr->row_ptr);
  free(csr->col_idx);
  csr->row_ptr = new_row_ptr;
  csr->col_idx = new_col_idx;
}

// Memory-efficient combined Reorder + Symmetrize + Build
static CSR_info build_reordered_symmetric_csr_from_arrays(int numrows,
                                                          int ncols, int *ir,
                                                          int *jc,
                                                          int64_t nir) {
  printf("Using Optimized Fast Path: Combined Reorder + Symmetrize...\n");

  // 1. Calculate Estimated Degrees (A + A^T)
  // We treat every entry in (ir, jc) as an undirected edge.
  // If (u, v) exists, we add +1 to u's degree and +1 to v's degree.
  // Duplicates (e.g. if symmetric input has u->v and v->u) will be handled by
  // deduplication later.

  int *degrees = calloc(numrows, sizeof(int));
  if (!degrees)
    return (CSR_info){0};

  cilk_for(int c = 0; c < ncols; c++) {
    for (int i = jc[c]; i < jc[c + 1]; i++) {
      int r = ir[i];
      if (r >= 0 && r < numrows) {
        __atomic_fetch_add(&degrees[r], 1, __ATOMIC_RELAXED);
        if (r != c) {
          // If square matrix, c is also a row index
          __atomic_fetch_add(&degrees[c], 1, __ATOMIC_RELAXED);
        }
      }
    }
  }

  // 2. Generate Permutation (Sort by Degree Descending)
  NodeDegree *nodes = malloc(numrows * sizeof(NodeDegree));
  cilk_for(int i = 0; i < numrows; i++) {
    nodes[i].id = i;
    nodes[i].degree = degrees[i];
  }

  // Sort nodes to determine permutation
  qsort(nodes, numrows, sizeof(NodeDegree), cmp_degree_desc);

  int *old_to_new = malloc(numrows * sizeof(int));
  int *new_to_old = malloc(numrows * sizeof(int));

  cilk_for(int i = 0; i < numrows; i++) {
    new_to_old[i] = nodes[i].id;
    old_to_new[nodes[i].id] = i;
  }

  // Calculate new row pointers based on permuted degrees
  // This is an UPPER BOUND on the edges (assuming all are unique).
  long long *new_row_ptr =
      aligned_alloc_wrapper((numrows + 1) * sizeof(long long));
  new_row_ptr[0] = 0;

  int *permuted_degrees = malloc(numrows * sizeof(int));
  cilk_for(int i = 0; i < numrows; i++) {
    permuted_degrees[i] =
        degrees[nodes[i].id]; // Degree of the node that will be at i
  }
  free(nodes);
  free(degrees); // Free old degrees

  // Prefix sum for row pointers
  for (int i = 0; i < numrows; i++) {
    new_row_ptr[i + 1] = new_row_ptr[i] + permuted_degrees[i];
  }
  free(permuted_degrees);

  long long max_nz = new_row_ptr[numrows];
  int *new_col_idx = aligned_alloc_wrapper(max_nz * sizeof(int));

  // 3. Fill Edges with Permutation
  long long *offsets = calloc(numrows, sizeof(long long));

  cilk_for(int c = 0; c < ncols; c++) {
    int new_c = old_to_new[c];
    for (int i = jc[c]; i < jc[c + 1]; i++) {
      int r = ir[i];
      if (r >= 0 && r < numrows) {
        int new_r = old_to_new[r];

        // Add mapping (new_r, new_c)
        long long dest_r =
            new_row_ptr[new_r] +
            __atomic_fetch_add(&offsets[new_r], 1, __ATOMIC_RELAXED);
        new_col_idx[dest_r] = new_c;

        if (new_r != new_c) {
          // Add mapping (new_c, new_r)
          long long dest_c =
              new_row_ptr[new_c] +
              __atomic_fetch_add(&offsets[new_c], 1, __ATOMIC_RELAXED);
          new_col_idx[dest_c] = new_r;
        }
      }
    }
  }
  free(offsets);
  free(old_to_new);
  // new_to_old is usually lost here, unless we wanted to save the permutation.
  // The current mat_to_csr doesn't save the permutation mapping to disk, so we
  // free it.
  free(new_to_old);

  // 4. Sort, Deduplicate, and Compact
  int *final_row_counts = calloc(numrows, sizeof(int));

  cilk_for(int i = 0; i < numrows; i++) {
    long long start = new_row_ptr[i];
    int count = new_row_ptr[i + 1] - start;
    final_row_counts[i] = sort_deduplicate_row(&new_col_idx[start], count);
  }

  // Final Row Pointers
  long long *final_row_ptr =
      aligned_alloc_wrapper((numrows + 1) * sizeof(long long));
  final_row_ptr[0] = 0;
  for (int i = 0; i < numrows; i++) {
    final_row_ptr[i + 1] = final_row_ptr[i] + final_row_counts[i];
  }

  long long final_nz = final_row_ptr[numrows];
  int *final_col_idx = aligned_alloc_wrapper(final_nz * sizeof(int));

  // Copy compacted
  cilk_for(int i = 0; i < numrows; i++) {
    long long src_start = new_row_ptr[i];
    long long dest_start = final_row_ptr[i];
    int count = final_row_counts[i];
    memcpy(&final_col_idx[dest_start], &new_col_idx[src_start],
           count * sizeof(int));
  }

  free(new_row_ptr);
  free(new_col_idx);
  free(final_row_counts);

  return (CSR_info){.nrows = numrows,
                    .nzcount = final_nz,
                    .row_ptr = final_row_ptr,
                    .col_idx = final_col_idx};
}

static char *csr_cache_filename(const char *matfile) {
  size_t l = strlen(matfile) + 8;
  char *out = malloc(l);
  if (!out)
    return NULL;
  snprintf(out, l, "%s.csrbin", matfile);
  return out;
}

static int write_csr_bin(const char *fname, const CSR_info *csr) {
  FILE *f = fopen(fname, "wb");
  if (!f)
    return -1;
  int64_t nrows = csr->nrows;
  int64_t nz = csr->nzcount;
  if (fwrite(&nrows, sizeof(int64_t), 1, f) != 1) {
    fclose(f);
    return -1;
  }
  if (fwrite(&nz, sizeof(int64_t), 1, f) != 1) {
    fclose(f);
    return -1;
  }
  if (fwrite(csr->row_ptr, sizeof(long long), (size_t)(nrows + 1), f) !=
      (size_t)(nrows + 1)) {
    fclose(f);
    return -1;
  }
  if (fwrite(csr->col_idx, sizeof(int), (size_t)nz, f) != (size_t)nz) {
    fclose(f);
    return -1;
  }
  fclose(f);
  return 0;
}

static CSR_info read_csr_bin(const char *fname) {
  CSR_info csr = {0, 0, NULL, NULL};
  FILE *f = fopen(fname, "rb");
  if (!f)
    return csr;
  int64_t nrows = 0, nz = 0;
  if (fread(&nrows, sizeof(int64_t), 1, f) != 1) {
    fclose(f);
    return csr;
  }
  if (fread(&nz, sizeof(int64_t), 1, f) != 1) {
    fclose(f);
    return csr;
  }
  csr.nrows = (int)nrows;
  csr.nzcount = (long long)nz;
  csr.row_ptr = (long long *)aligned_alloc_wrapper((size_t)(csr.nrows + 1) *
                                                   sizeof(long long));
  csr.col_idx = (int *)aligned_alloc_wrapper((size_t)csr.nzcount * sizeof(int));
  if (!csr.row_ptr || !csr.col_idx) {
    fclose(f);
    clean_csr_vectors(csr);
    return (CSR_info){0, 0, NULL, NULL};
  }
  if (fread(csr.row_ptr, sizeof(long long), (size_t)(csr.nrows + 1), f) !=
      (size_t)(csr.nrows + 1)) {
    fclose(f);
    clean_csr_vectors(csr);
    return (CSR_info){0, 0, NULL, NULL};
  }
  if (fread(csr.col_idx, sizeof(int), (size_t)csr.nzcount, f) !=
      (size_t)csr.nzcount) {
    fclose(f);
    clean_csr_vectors(csr);
    return (CSR_info){0, 0, NULL, NULL};
  }
  fclose(f);
  return csr;
}

// HDF5 Reader for Memory Efficiency
// Must link with -lhdf5
#include <hdf5.h>

// HDF5 Chunked Reading Helper
typedef struct {
  hid_t dset_id;
  hid_t type_id;
  size_t type_size;
  hsize_t total_elems;
  hsize_t current_idx;
  size_t buffer_size; // Elements per chunk
  void *buffer;
  hsize_t buffer_end_idx; // Exclusive
  hsize_t buffer_curr_local_idx;
} H5Streamer;

static int h5_streamer_init(H5Streamer *s, hid_t dset_id, hid_t type_id,
                            size_t buffer_size) {
  s->dset_id = dset_id;
  s->type_id = type_id;
  s->type_size = H5Tget_size(type_id);

  hid_t space_id = H5Dget_space(dset_id);
  s->total_elems = H5Sget_simple_extent_npoints(space_id);
  H5Sclose(space_id);

  s->buffer_size = buffer_size;
  s->current_idx = 0;

  s->buffer = malloc(buffer_size * s->type_size);
  if (!s->buffer)
    return -1;

  // Initial fetch
  s->buffer_end_idx = 0;
  s->buffer_curr_local_idx = 0;
  return 0;
}

static void h5_streamer_close(H5Streamer *s) {
  if (s->buffer)
    free(s->buffer);
}

// Fetch next element (int). Buffers behind the scenes.
// Returns -1 on EOF or error (assuming all valid values are >= 0)
// This is specific for 'ir' (int) streaming.
static inline int h5_streamer_next_int(H5Streamer *s) {
  // Check if we need to refill buffer
  if (s->buffer_curr_local_idx >= (s->buffer_end_idx - s->current_idx)) {
    // Refill
    if (s->current_idx + (s->buffer_end_idx - s->current_idx) >=
        s->total_elems) {
      return -1; // EOF
    }

    s->current_idx = s->buffer_end_idx;
    hsize_t count = s->buffer_size;
    if (s->current_idx + count > s->total_elems) {
      count = s->total_elems - s->current_idx;
    }

    // Select hyperslab
    hid_t memspace = H5Screate_simple(1, &count, NULL);
    hid_t filespace = H5Dget_space(s->dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, &s->current_idx, NULL,
                        &count, NULL);

    if (H5Dread(s->dset_id, s->type_id, memspace, filespace, H5P_DEFAULT,
                s->buffer) < 0) {
      H5Sclose(memspace);
      H5Sclose(filespace);
      return -2; // Error
    }

    H5Sclose(memspace);
    H5Sclose(filespace);

    s->buffer_end_idx = s->current_idx + count;
    s->buffer_curr_local_idx = 0;
  }

  int val = ((int *)s->buffer)[s->buffer_curr_local_idx++];
  return val;
}

static CSR_info try_read_hdf5_chunked(const char *fname) {
  H5Eset_auto(H5E_DEFAULT, NULL, NULL);
  if (!H5Fis_hdf5(fname))
    return (CSR_info){0};

  printf("HDF5: Detected MAT v7.3 file. Using Two-Pass Chunked Streamer...\n");

  hid_t file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0)
    return (CSR_info){0};

  hid_t group_id = H5Gopen(file_id, "Problem", H5P_DEFAULT);
  if (group_id < 0) {
    H5Fclose(file_id);
    return (CSR_info){0};
  }

  hid_t a_id = H5Gopen(group_id, "A", H5P_DEFAULT);
  if (a_id < 0) {
    H5Gclose(group_id);
    H5Fclose(file_id);
    return (CSR_info){0};
  }

  hid_t ir_dset = H5Dopen(a_id, "ir", H5P_DEFAULT);
  hid_t jc_dset = H5Dopen(a_id, "jc", H5P_DEFAULT);

  if (ir_dset < 0 || jc_dset < 0) {
    if (ir_dset >= 0)
      H5Dclose(ir_dset);
    if (jc_dset >= 0)
      H5Dclose(jc_dset);
    H5Gclose(a_id);
    H5Gclose(group_id);
    H5Fclose(file_id);
    return (CSR_info){0};
  }

  // Get Sizes
  hid_t ir_space = H5Dget_space(ir_dset);
  hid_t jc_space = H5Dget_space(jc_dset);

  hsize_t nnz = H5Sget_simple_extent_npoints(ir_space);
  hsize_t n_col_ptrs = H5Sget_simple_extent_npoints(jc_space);
  int nrows = (int)(n_col_ptrs - 1);
  int ncols = nrows; // Assume square symmetric

  printf("HDF5 Streaming: Rows/Cols: %d, NNZ: %llu\n", nrows,
         (unsigned long long)nnz);

  // Load JC fully (pointer array is usually manageable, ~520MB for Friendster)
  // If this is too large, we would need 2-level streaming, but let's assume it
  // fits.
  printf("HDF5: Loading Col Ptrs (jc)...\n");
  long long *jc = aligned_alloc_wrapper(n_col_ptrs * sizeof(long long));
  if (H5Dread(jc_dset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, jc) <
      0) {
    free(jc);
    return (CSR_info){0};
  }

  // --- PASS 1: Calculate Degrees ---
  printf("HDF5: Pass 1 - Calculating Degrees (Streaming ir)...\n");

  int *degrees = calloc(nrows, sizeof(int));
  if (!degrees) {
    free(jc);
    return (CSR_info){0};
  }

  H5Streamer ir_streamer;
  // 128MB chunk for ir (int) -> 32M integers
  if (h5_streamer_init(&ir_streamer, ir_dset, H5T_NATIVE_INT,
                       32 * 1024 * 1024) != 0) {
    free(jc);
    free(degrees);
    return (CSR_info){0};
  }

  // Stream through all columns
  for (int c = 0; c < ncols; c++) {
    long long start = jc[c];
    long long end = jc[c + 1];
    long long count = end - start;

    for (long long k = 0; k < count; k++) {
      int r = h5_streamer_next_int(&ir_streamer);
      if (r < 0)
        break; // Error or EOF premature

      degrees[r]++;
      if (r != c)
        degrees[c]++;
    }
  }

  h5_streamer_close(&ir_streamer); // Reset streamer

  // --- Sort & Permute ---
  printf("HDF5: Calculating Permutation...\n");
  NodeDegree *nodes = malloc(nrows * sizeof(NodeDegree));
  cilk_for(int i = 0; i < nrows; i++) {
    nodes[i].id = i;
    nodes[i].degree = degrees[i];
  }
  qsort(nodes, nrows, sizeof(NodeDegree), cmp_degree_desc);

  int *old_to_new = malloc(nrows * sizeof(int));
  cilk_for(int i = 0; i < nrows; i++) { old_to_new[nodes[i].id] = i; }

  long long *new_row_ptr =
      aligned_alloc_wrapper((nrows + 1) * sizeof(long long));
  new_row_ptr[0] = 0;
  int *permuted_degrees = malloc(nrows * sizeof(int));
  cilk_for(int i = 0; i < nrows; i++) {
    permuted_degrees[i] = degrees[nodes[i].id];
  }
  free(nodes);
  free(degrees);

  for (int i = 0; i < nrows; i++) {
    new_row_ptr[i + 1] = new_row_ptr[i] + permuted_degrees[i];
  }
  free(permuted_degrees);

  long long final_nz = new_row_ptr[nrows];
  int *new_col_idx = aligned_alloc_wrapper(final_nz * sizeof(int));
  long long *offsets = calloc(nrows, sizeof(long long));

  // --- PASS 2: Build Graph ---
  printf("HDF5: Pass 2 - Building Graph (Streaming ir)...\n");

  // Re-init streamer at start
  h5_streamer_init(&ir_streamer, ir_dset, H5T_NATIVE_INT, 32 * 1024 * 1024);

  // NOTE: HDF5 Streaming is inherently forcing serial execution of the
  // 'fetching' part. We can't parallelize the loop over 'c' easily without
  // multiple independent readers/cursors. However, the bottleneck is IO/Memory,
  // so serial reading is fine. We can still write to 'new_col_idx' in parallel
  // if we buffered the reads, but standard loop is fine.

  for (int c = 0; c < ncols; c++) {
    long long start = jc[c];
    long long end = jc[c + 1];
    long long count = end - start;
    int new_c = old_to_new[c];

    for (long long k = 0; k < count; k++) {
      int r = h5_streamer_next_int(&ir_streamer);
      if (r >= 0 && r < nrows) {
        int new_r = old_to_new[r];

        // Add (new_r, new_c)
        long long dest_r = new_row_ptr[new_r] + offsets[new_r]++;
        new_col_idx[dest_r] = new_c;

        if (new_r != new_c) {
          long long dest_c = new_row_ptr[new_c] + offsets[new_c]++;
          new_col_idx[dest_c] = new_r;
        }
      }
    }
  }
  h5_streamer_close(&ir_streamer);
  free(offsets);
  free(old_to_new);
  free(jc); // Done with raw JC

  // --- Sort & Deduplicate ---
  printf("HDF5: Sorting and Compacting...\n");
  int *final_row_counts = calloc(nrows, sizeof(int));

  cilk_for(int i = 0; i < nrows; i++) {
    long long start = new_row_ptr[i];
    int count = new_row_ptr[i + 1] - start;
    final_row_counts[i] = sort_deduplicate_row(&new_col_idx[start], count);
  }

  long long *final_row_ptr =
      aligned_alloc_wrapper((nrows + 1) * sizeof(long long));
  final_row_ptr[0] = 0;
  for (int i = 0; i < nrows; i++) {
    final_row_ptr[i + 1] = final_row_ptr[i] + final_row_counts[i];
  }

  final_nz = final_row_ptr[nrows];
  int *final_col_idx = aligned_alloc_wrapper(final_nz * sizeof(int));

  cilk_for(int i = 0; i < nrows; i++) {
    long long src_start = new_row_ptr[i];
    long long dest_start = final_row_ptr[i];
    int count = final_row_counts[i];
    memcpy(&final_col_idx[dest_start], &new_col_idx[src_start],
           count * sizeof(int));
  }

  free(new_row_ptr);
  free(new_col_idx);
  free(final_row_counts);

  H5Sclose(ir_space);
  H5Sclose(jc_space);
  H5Dclose(ir_dset);
  H5Dclose(jc_dset);
  H5Gclose(a_id);
  H5Gclose(group_id);
  H5Fclose(file_id);

  return (CSR_info){.nrows = nrows,
                    .nzcount = final_nz,
                    .row_ptr = final_row_ptr,
                    .col_idx = final_col_idx};
}

CSR_info mat_to_csr(char *file_name) {
  /* Check cache first */
  char *cache = csr_cache_filename(file_name);
  if (cache) {
    struct stat st;
    if (stat(cache, &st) == 0) {
      CSR_info csr = read_csr_bin(cache);
      if (csr.row_ptr && csr.col_idx) {
        printf("Loaded CSR from cache %s\n", cache);
        free(cache);
        return csr;
      }
      /* read failed, continue to build */
    }

    // Try HDF5 Reader for v7.3 MAT files (Low Memory Path)
    CSR_info h5_csr = try_read_hdf5_chunked(file_name);
    if (h5_csr.nzcount > 0) {
      printf("Successfully loaded via HDF5 Two-Pass Streaming.\n");
      // The chunked loader already does reordering and symmetrization
      // So we just return it.

      // Write to cache
      if (write_csr_bin(cache, &h5_csr) == 0) {
        printf("Wrote CSR cache %s\n", cache);
      }
      free(cache);
      return h5_csr;
    }
    // If HDF5 failed, it returns empty and we fall through to MATIO
  }

  // HDF5 failed (maybe not v7.3), fall back to MATIO
  free(cache);

  // Re-alloc cache name for MATIO path
  cache = csr_cache_filename(file_name);

  // Open .mat file
  mat_t *matfp = Mat_Open(file_name, MAT_ACC_RDONLY);
  if (!matfp) {
    fprintf(stderr, "MATIO->ERROR: cannot open %s file \n", file_name);
    return (CSR_info){1, 1, NULL, NULL};
  }

  // Read the sparse matrix  with variable name "A" inside struct "Problem"
  matvar_t *projectvar = Mat_VarRead(matfp, "Problem");
  if (!projectvar) {
    fprintf(stderr, "MATIO->ERROR:'Problem' struct missing in %s\n", file_name);
    Mat_Close(matfp);
    return (CSR_info){1, 1, NULL, NULL};
  }

  matvar_t *matvar = Mat_VarGetStructFieldByName(projectvar, "A", 0);
  if (!matvar) {
    fprintf(stderr,
            "MATIO->ERROR: 'A' matrix missing in struct 'Problem' inside %s\n",
            file_name);
    Mat_VarFree(projectvar);
    Mat_Close(matfp);
    return (CSR_info){1, 1, NULL, NULL};
  }

  if (matvar->class_type != MAT_C_SPARSE) {
    fprintf(stderr, "MATIO->ERROR: matrix read in %s is not sparse\n",
            file_name);
    Mat_VarFree(projectvar);
    Mat_Close(matfp);
    return (CSR_info){1, 1, NULL, NULL};
  }

  mat_sparse_t *sparse = (mat_sparse_t *)matvar->data;
  int numrows = matvar->dims[0];
  int ncols = matvar->dims[1];
  int *row_ind = (int *)sparse->ir;
  int *col_ptr = (int *)sparse->jc;

  // MEMORY OPTIMIZATION: Free the values array immediately
  // We only need the structure (ir, jc), not the values (data).
  // matio allocates data as double/complex. For 2.7B edges, this is ~21GB we
  // don't need.
  if (sparse->data != NULL) {
    free(sparse->data);
    sparse->data = NULL; // Prevent double free by Mat_VarFree
    printf("Freed sparse matrix values to save memory.\n");
  }

  // Check env vars early
  char *env_sym = getenv("BFS_ASSUME_SYMMETRIC");
  int assume_symmetric = (env_sym && atoi(env_sym) == 1);

  // OPTIMIZATION: Direct Cache Generation for Symmetric Graphs
  // If we assume symmetry (CSC=CSR), we can write the .csrbin DIRECTLY from
  // matio arrays without allocating the huge CSR struct in memory
  // simultaneously.
  if (assume_symmetric && cache) {
    printf("BFS_ASSUME_SYMMETRIC=1: Generating Cache Directly from Input to "
           "save RAM...\n");

    FILE *f = fopen(cache, "wb");
    if (f) {
      int64_t nrows_64 = numrows;
      // nzcount in CSC is stored in jc[ncols] (or roughly, compressed size)
      // actually sparse->nir is number of nonzeros
      int64_t nz_64 = sparse->nir;

      fwrite(&nrows_64, sizeof(int64_t), 1, f);
      fwrite(&nz_64, sizeof(int64_t), 1, f);

// Write row_ptr (convert from col_ptr ints to long long)
// Since CSC=CSR for symmetric, col_ptr (int) becomes row_ptr (long long)
// We buffer the writes to avoid allocating full array
#define WRITE_BUF_SIZE 8192
      long long *buf_ptr = malloc(WRITE_BUF_SIZE * sizeof(long long));
      if (buf_ptr) {
        int idx = 0;
        for (int i = 0; i <= numrows; i++) { // numrows=ncols for symmetric
          buf_ptr[idx++] = (long long)col_ptr[i];
          if (idx == WRITE_BUF_SIZE) {
            fwrite(buf_ptr, sizeof(long long), idx, f);
            idx = 0;
          }
        }
        if (idx > 0)
          fwrite(buf_ptr, sizeof(long long), idx, f);
        free(buf_ptr);
      }

      // Write col_idx (copy from row_ind ints)
      // row_ind is int, col_idx is int. Direct write!
      fwrite(row_ind, sizeof(int), (size_t)nz_64, f);

      fclose(f);
      printf("Direct Cache Generation Complete. Reloading efficiently...\n");

      // Clean up Matio Source Memory completely
      Mat_VarFree(projectvar);
      Mat_Close(matfp);
      free(cache);

      // Now reload using efficient reader
      // Recalculate cache name since we freed it (oops, let's just re-call
      // helper or assume same) Actually read_csr_bin needs filename.
      char *reload_cache = csr_cache_filename(file_name);
      CSR_info reloaded = read_csr_bin(reload_cache);
      free(reload_cache);

      // Skip the rest of standard logic
      return reloaded;
    }
    // If file open failed, fall back to standard path
  }

  // Optimize: Use Fast Path if env vars allow (Default ON)
  CSR_info csr = {0, 0, NULL, NULL};
  int skip_reorder =
      (getenv("BFS_SKIP_REORDER") && atoi(getenv("BFS_SKIP_REORDER")) == 1);
  int skip_symmetric = (getenv("BFS_ASSUME_SYMMETRIC") &&
                        atoi(getenv("BFS_ASSUME_SYMMETRIC")) == 1);

  if (!skip_reorder && !skip_symmetric) {
    // Fast path: Unified Symmetrization + Reordering
    csr = build_reordered_symmetric_csr_from_arrays(numrows, ncols, row_ind,
                                                    col_ptr, sparse->nir);

    // Cleanup Matio
    Mat_VarFree(projectvar);
    Mat_Close(matfp);
  } else {
    // Legacy Path (Slow)

    // ... (Original logic kept as fallback) ...

    csr.row_ptr = (long long *)aligned_alloc_wrapper((size_t)(numrows + 1) *
                                                     sizeof(long long));
    if (csr.row_ptr)
      memset(csr.row_ptr, 0, (size_t)(numrows + 1) * sizeof(long long));

    // Parallel count nonzeros per row (atomic increments)
    int *row_counts = (int *)calloc((size_t)numrows, sizeof(int));
    if (!row_counts) {
      Mat_VarFree(projectvar);
      Mat_Close(matfp);
      return (CSR_info){1, 1, NULL, NULL};
    }

    cilk_for(int c = 0; c < ncols; ++c) {
      for (int i = col_ptr[c]; i < col_ptr[c + 1]; ++i) {
        int r = row_ind[i];
        if (r >= 0 && r < numrows) {
          __atomic_fetch_add(&row_counts[r], 1, __ATOMIC_RELAXED);
        }
      }
    }

    // Build row_ptr via prefix sum (sequential)
    csr.row_ptr[0] = 0;
    for (int i = 0; i < numrows; ++i) {
      csr.row_ptr[i + 1] = csr.row_ptr[i] + row_counts[i];
    }

    // Define non-zeros and rows count
    csr.nzcount = csr.row_ptr[numrows];
    csr.nrows = numrows;

    printf("Number of nonzeros: %lld\n", csr.nzcount);

    // Define col_idx
    csr.col_idx = (int *)aligned_alloc_wrapper(csr.nzcount * sizeof(int));

    long long *pos = (long long *)malloc((size_t)numrows * sizeof(long long));
    if (!pos) {
      free(row_counts);
      Mat_VarFree(projectvar);
      Mat_Close(matfp);
      clean_csr_vectors(csr);
      return (CSR_info){1, 1, NULL, NULL};
    }
    for (int i = 0; i < numrows; ++i)
      pos[i] = csr.row_ptr[i];

    // Fill col_idx in parallel; reserve positions via atomic fetch-add
    cilk_for(int c = 0; c < ncols; ++c) {
      for (int j = col_ptr[c]; j < col_ptr[c + 1]; ++j) {
        int r = row_ind[j];
        if (r < 0 || r >= numrows)
          continue;
        long long dest = __atomic_fetch_add(&pos[r], 1, __ATOMIC_RELAXED);
        csr.col_idx[dest] = c;
      }
    }

    free(row_counts);
    free(pos);

    // Free MATIO memory immediately to reduce peak usage
    Mat_VarFree(projectvar);
    Mat_Close(matfp);

    // Sort rows to enable binary search and efficient processing
    cilk_for(int i = 0; i < numrows; i++) {
      int start = csr.row_ptr[i];
      int count = csr.row_ptr[i + 1] - start;
      sort_deduplicate_row(&csr.col_idx[start], count); // Sort only (no dupes)
    }

    // Check symmetry and symmetrize if needed
    // The declaration of env_sym and assume_symmetric has been moved up.

    if (assume_symmetric) {
      printf("BFS_ASSUME_SYMMETRIC=1: Skipping symmetry check and "
             "symmetrization.\n");
    } else {
      printf("Checking symmetry...\n");
      if (!is_symmetric(&csr)) {
        printf("Graph not symmetric. Symmetrizing...\n");
        symmetrize_graph(&csr);
        printf("Symmetrization complete. New edges: %lld\n", csr.nzcount);
      } else {
        printf("Graph is symmetric.\n");
      }
    }

    // Optimization: Graph Reordering (Degree Descending)
    char *env_reorder = getenv("BFS_SKIP_REORDER");
    if (env_reorder && atoi(env_reorder) == 1) {
      printf("BFS_SKIP_REORDER=1: Skipping graph reordering.\n");
    } else {
      printf("Reordering graph (Degree Descending)...\n");
      reorder_csr_by_degree(&csr);
      printf("Graph reordering complete.\n");
    }
  }

  printf("Your .mat to CSR format in .bin files conversion complete.\n");

  /* Write cache for next time (best-effort) */
  cache = csr_cache_filename(file_name);
  if (cache) {
    if (write_csr_bin(cache, &csr) == 0) {
      printf("Wrote CSR cache %s\n", cache);
    }
    free(cache);
  }

  return csr;
}

// Compute edge-balanced partition
// Returns an array of size (size + 1) containing start row for each rank
// vtx_dist[0] = 0, vtx_dist[size] = N
int *compute_balanced_partition(const char *fname, int size, int *out_n) {
  FILE *f = fopen(fname, "rb");
  if (!f)
    return NULL;

  int64_t global_nrows = 0, global_nz = 0;
  if (fread(&global_nrows, sizeof(int64_t), 1, f) != 1 ||
      fread(&global_nz, sizeof(int64_t), 1, f) != 1) {
    fclose(f);
    return NULL;
  }

  if (out_n)
    *out_n = (int)global_nrows;

  long long *row_ptr = malloc(((int)global_nrows + 1) * sizeof(long long));
  if (!row_ptr) {
    fclose(f);
    return NULL;
  }

  fseek(f, 16, SEEK_SET);
  if (fread(row_ptr, sizeof(long long), (int)global_nrows + 1, f) !=
      (size_t)(global_nrows + 1)) {
    free(row_ptr);
    fclose(f);
    return NULL;
  }
  fclose(f);

  int *vtx_dist = malloc((size + 1) * sizeof(int));
  vtx_dist[0] = 0;
  vtx_dist[size] = (int)global_nrows;

  long long total_nz = row_ptr[global_nrows];
  long long target_nz = total_nz / size;

  int current_rank = 1;
  long long current_target = target_nz;

  for (int i = 0; i < global_nrows; i++) {
    if (current_rank >= size)
      break;
    if (row_ptr[i] >= current_target) {
      vtx_dist[current_rank] = i;
      current_rank++;
      current_target += target_nz;
    }
  }
  while (current_rank < size) {
    vtx_dist[current_rank] = (int)global_nrows;
    current_rank++;
  }

  free(row_ptr);
  return vtx_dist;
}

// Read only a part of the CSR binary file for distributed loading
CSR_info read_csr_bin_part(const char *fname, int start_row, int end_row,
                           int *out_global_n) {
  CSR_info csr = {0, 0, NULL, NULL};
  FILE *f = fopen(fname, "rb");
  if (!f)
    return csr;

  int64_t global_nrows = 0, global_nz = 0;
  if (fread(&global_nrows, sizeof(int64_t), 1, f) != 1 ||
      fread(&global_nz, sizeof(int64_t), 1, f) != 1) {
    fclose(f);
    return csr;
  }

  if (out_global_n)
    *out_global_n = (int)global_nrows;

  int my_start_row = start_row;
  int my_end_row = end_row;
  int my_nrows = my_end_row - my_start_row;

  csr.nrows = my_nrows;
  // Use malloc for safety instead of aligned_alloc
  csr.row_ptr = malloc((my_nrows + 1) * sizeof(long long));
  if (!csr.row_ptr) {
    fclose(f);
    return csr;
  }

  long row_ptr_offset = 16 + (long)my_start_row * sizeof(long long);

  fseek(f, row_ptr_offset, SEEK_SET);
  if (fread(csr.row_ptr, sizeof(long long), my_nrows + 1, f) !=
      (size_t)(my_nrows + 1)) {
    fclose(f);
    free(csr.row_ptr);
    return (CSR_info){0};
  }

  long long start_edge_idx = csr.row_ptr[0];
  long long end_edge_idx = csr.row_ptr[my_nrows];
  long long my_nz = end_edge_idx - start_edge_idx;
  csr.nzcount = my_nz;

  for (int i = 0; i <= my_nrows; i++) {
    csr.row_ptr[i] -= start_edge_idx;
  }

  // Use malloc for column indices as well
  csr.col_idx = malloc((size_t)my_nz * sizeof(int));
  if (!csr.col_idx) {
    fclose(f);
    free(csr.row_ptr);
    return (CSR_info){0};
  }

  long col_idx_start = 16 + (long)(global_nrows + 1) * sizeof(long long);
  long my_col_offset = col_idx_start + (long)start_edge_idx * sizeof(int);

  fseek(f, my_col_offset, SEEK_SET);
  if (fread(csr.col_idx, sizeof(int), my_nz, f) != (size_t)my_nz) {
    fclose(f);
    free(csr.row_ptr);
    free(csr.col_idx);
    return (CSR_info){0};
  }

  fclose(f);
  return csr;
}
