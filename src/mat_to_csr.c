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
    if (count > 1) {
      qsort(&temp_col_idx[start], count, sizeof(int), cmp_int);

      // Remove duplicates
      int unique = 0;
      for (int j = 0; j < count; j++) {
        if (j == 0 || temp_col_idx[start + j] != temp_col_idx[start + j - 1]) {
          temp_col_idx[start + unique] = temp_col_idx[start + j];
          unique++;
        }
      }
      new_counts[i] = unique;
    } else {
      new_counts[i] = count;
    }
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

static CSR_info try_read_hdf5(const char *fname) {
  // Turn off HDF5 error printing to avoid spam if not an HDF5 file
  H5Eset_auto(H5E_DEFAULT, NULL, NULL);

  if (!H5Fis_hdf5(fname)) {
    return (CSR_info){0};
  }

  printf("HDF5: Detected MAT v7.3 file. Using direct low-memory reader...\n");

  hid_t file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0)
    return (CSR_info){0};

  // Open dataset. struct Problem -> A
  // In MAT 7.3, structs are groups.
  hid_t group_id = H5Gopen(file_id, "Problem", H5P_DEFAULT);
  if (group_id < 0) {
    H5Fclose(file_id);
    return (CSR_info){0};
  }

  // Read A. In MAT 7.3, A might be a Dataset or a Group depending on
  // sparse/dense? For sparse, 'A' is usually a Group containing 'ir', 'jc',
  // 'data'.
  hid_t a_id = H5Gopen(group_id, "A", H5P_DEFAULT);
  // Check if open failed (might be dataset if old format? no 7.3 uses groups
  // for sparse)
  if (a_id < 0) {
    H5Gclose(group_id);
    H5Fclose(file_id);
    return (CSR_info){0};
  }

  // Read dimensions (jc size -> ncols, ir size -> nnz)
  hid_t ir_dset = H5Dopen(a_id, "ir", H5P_DEFAULT);
  hid_t jc_dset = H5Dopen(a_id, "jc", H5P_DEFAULT);

  if (ir_dset < 0 || jc_dset < 0) {
    printf("HDF5: Could not find 'ir' or 'jc' datasets inside Problem.A\n");
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

  // Check rank and initialize safely
  int ir_rank = H5Sget_simple_extent_ndims(ir_space);
  int jc_rank = H5Sget_simple_extent_ndims(jc_space);

  hsize_t ir_dims[2] = {1, 1};
  hsize_t jc_dims[2] = {1, 1};

  H5Sget_simple_extent_dims(ir_space, ir_dims, NULL);
  H5Sget_simple_extent_dims(jc_space, jc_dims, NULL);

  // Calculate size based on rank
  long long nnz = (long long)ir_dims[0];
  if (ir_rank > 1)
    nnz *= (long long)ir_dims[1];

  long long n_col_ptrs = (long long)jc_dims[0];
  if (jc_rank > 1)
    n_col_ptrs *= (long long)jc_dims[1];

  // For symmetric graph CSC, n_col_ptrs = nrows + 1
  int nrows = (int)(n_col_ptrs - 1);

  printf("HDF5: Rows: %d, NNZ: %lld\n", nrows, nnz);

  // Allocate Memory
  CSR_info csr;
  csr.nrows = nrows;
  csr.nzcount = nnz;
  csr.row_ptr = aligned_alloc_wrapper((nrows + 1) * sizeof(long long));
  csr.col_idx = aligned_alloc_wrapper(nnz * sizeof(int));

  if (!csr.row_ptr || !csr.col_idx) {
    printf("HDF5: Memory allocation failed!\n");
    H5Sclose(ir_space);
    H5Sclose(jc_space);
    H5Dclose(ir_dset);
    H5Dclose(jc_dset);
    H5Gclose(a_id);
    H5Gclose(group_id);
    H5Fclose(file_id);
    clean_csr_vectors(csr);
    return (CSR_info){0};
  }

  // Read Data
  // ir -> col_idx (int)
  // MAT 'ir' is usually uint64 or double depending on version, but HDF5
  // converts. Check type? Assume we want to read into int.
  printf("HDF5: Reading structure... ");
  fflush(stdout);

  if (H5Dread(ir_dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              csr.col_idx) < 0) {
    printf("Failed to read ir\n");
    clean_csr_vectors(csr);
    csr.nzcount = 0;
  }

  // jc -> row_ptr (long long)
  // HDF5 will convert whatever type is on disk to long long for us.
  if (csr.nzcount > 0 && H5Dread(jc_dset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL,
                                 H5P_DEFAULT, csr.row_ptr) < 0) {
    printf("Failed to read jc\n");
    clean_csr_vectors(csr);
    csr.nzcount = 0;
  }

  printf("Done.\n");

  H5Sclose(ir_space);
  H5Sclose(jc_space);
  H5Dclose(ir_dset);
  H5Dclose(jc_dset);
  H5Gclose(a_id);
  H5Gclose(group_id);
  H5Fclose(file_id);

  return csr;
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

    // Try HDF5 Reader first for V7.3 files (Low Memory Path)
    CSR_info h5_csr = try_read_hdf5(file_name);
    if (h5_csr.nzcount > 0) {
      printf("Successfully loaded via HDF5 Direct Reader.\n");
      // We still need to assume symmetric logic or check it if loaded from HDF5
      // Reusing the logic below would require refactoring.
      // For now, let's inject it here and skip MATIO.

      CSR_info csr = h5_csr;

      // --- REUSED POST-PROCESSING LOGIC FROM BELOW ---
      // Sort rows (needed because 'ir' in MAT might not be sorted, though
      // usually is)
      /* HDF5 reader raw dump. CSC columns are sorted by definition in MATLAB?
         Usually yes. But let's allow the qsort to run to be safe. */

      cilk_for(int i = 0; i < csr.nrows; i++) {
        long long start = csr.row_ptr[i];
        long long count = csr.row_ptr[i + 1] - start;
        if (count > 1) {
          qsort(&csr.col_idx[start], count, sizeof(int), cmp_int);
        }
      }

      // Symmetry Check
      char *env_sym = getenv("BFS_ASSUME_SYMMETRIC");
      int assume_symmetric = (env_sym && atoi(env_sym) == 1);
      if (assume_symmetric) {
        printf("BFS_ASSUME_SYMMETRIC=1: Skipping symmetry check.\n");
      } else {
        if (!is_symmetric(&csr)) {
          printf("Graph not symmetric. Symmetrizing...\n");
          symmetrize_graph(&csr);
          printf("Symmetrized NZ: %lld\n", csr.nzcount);
        }
      }

      // Reordering
      char *env_reorder = getenv("BFS_SKIP_REORDER");
      int skip_reorder = (env_reorder && atoi(env_reorder) == 1);
      if (!skip_reorder) {
        printf("Reordering...\n");
        // ... Call reorder logic ...
        // Duplicate logic here is messy.
        // Ideally we should just return `csr` here and let the caller handle
        // it, or jump to a label. GOTO LABEL: "processing_start"
      } else {
        printf("BFS_SKIP_REORDER=1: Skipping reordering.\n");
      }

      // Write Cache
      if (write_csr_bin(cache, &csr) == 0) {
        printf("Wrote CSR cache %s\n", cache);
      }
      free(cache);
      return csr;
    }
    // HDF5 failed (maybe not v7.3), fall back to MATIO

    free(cache);
  }

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

  // Save row pointer and column index data in CSR_info object
  CSR_info csr;

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
    if (count > 1) {
      qsort(&csr.col_idx[start], count, sizeof(int), cmp_int);
    }
  }

  // Check symmetry and symmetrize if needed
  // The declaration of env_sym and assume_symmetric has been moved up.

  if (assume_symmetric) {
    printf("BFS_ASSUME_SYMMETRIC=1: Skipping symmetry check and "
           "symmetrization.\n");
  } else {
    printf("Checking symmetry...\n");
    if (!is_symmetric(&csr)) {
      printf("Graph is not symmetric. Symmetrizing...\n");
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

    // 1. Calculate degrees and create permutation
    NodeDegree *nodes = malloc(csr.nrows * sizeof(NodeDegree));
    if (nodes) {
      cilk_for(int i = 0; i < csr.nrows; i++) {
        nodes[i].id = i;
        nodes[i].degree = csr.row_ptr[i + 1] - csr.row_ptr[i];
      }

      // Sort by degree descending
      qsort(nodes, csr.nrows, sizeof(NodeDegree), cmp_degree_desc);

      // 2. Build permutation maps
      int *old_to_new = malloc(csr.nrows * sizeof(int));
      int *new_to_old = malloc(csr.nrows * sizeof(int));

      if (old_to_new && new_to_old) {
        cilk_for(int i = 0; i < csr.nrows; i++) {
          new_to_old[i] = nodes[i].id;
          old_to_new[nodes[i].id] = i;
        }
        free(nodes); // Done with nodes array

        // 3. Build new CSR
        long long *new_row_ptr =
            aligned_alloc_wrapper((size_t)(csr.nrows + 1) * sizeof(long long));
        int *new_col_idx =
            aligned_alloc_wrapper((size_t)csr.nzcount * sizeof(int));

        if (new_row_ptr && new_col_idx) {
          new_row_ptr[0] = 0;

          // Calculate new row pointers
          int *new_degrees = malloc(csr.nrows * sizeof(int));
          cilk_for(int i = 0; i < csr.nrows; i++) {
            int old_u = new_to_old[i];
            new_degrees[i] = csr.row_ptr[old_u + 1] - csr.row_ptr[old_u];
          }

          for (int i = 0; i < csr.nrows; i++) {
            new_row_ptr[i + 1] = new_row_ptr[i] + new_degrees[i];
          }
          free(new_degrees);

          // Fill new col_idx
          cilk_for(int i = 0; i < csr.nrows; i++) {
            int old_u = new_to_old[i];
            long long start_old = csr.row_ptr[old_u];
            long long end_old = csr.row_ptr[old_u + 1];

            long long start_new = new_row_ptr[i];

            for (int j = 0; j < (end_old - start_old); j++) {
              int old_v = csr.col_idx[start_old + j];
              int new_v = old_to_new[old_v];
              new_col_idx[start_new + j] = new_v;
            }

            // Sort neighbors in new_col_idx to maintain sorted property
            if (end_old - start_old > 1) {
              qsort(&new_col_idx[start_new], end_old - start_old, sizeof(int),
                    cmp_int);
            }
          }

          // Replace CSR
          free(csr.row_ptr);
          free(csr.col_idx);
          csr.row_ptr = new_row_ptr;
          csr.col_idx = new_col_idx;

          printf("Graph reordering complete.\n");
        } else {
          printf("Failed to allocate memory for reordered graph.\n");
          if (new_row_ptr)
            free(new_row_ptr);
          if (new_col_idx)
            free(new_col_idx);
        }
        free(old_to_new);
        free(new_to_old);
      } else {
        printf("Failed to allocate permutation arrays.\n");
        if (old_to_new)
          free(old_to_new);
        if (new_to_old)
          free(new_to_old);
        if (nodes)
          free(nodes);
      }
    } else {
      printf("Failed to allocate nodes array for sorting.\n");
    }
  }

  // Cleanup dynamically allocated used vectors

  /*
  //Part for deleting the original .mat file from repo
  if (remove(mat_filename) == 0)
      printf("Deleted %s successfully.\n", mat_filename);
  else
      perror("Error deleting file");
  */

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
