#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <matio.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

#ifdef __opencilk
#include <cilk/cilk.h>
#else
#define cilk_for for
#define cilk_spawn
#define cilk_sync
#endif

#include "mat_to_csr.h"

// Helper for aligned allocation (64-byte alignment for AVX-512/Cache lines)
void* aligned_alloc_wrapper(size_t size) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, size) != 0) {
        return NULL;
    }
    return ptr;
}

void clean_csr_vectors(CSR_info vecs){
    free(vecs.row_ptr);
    free(vecs.col_idx);
}


static char *csr_cache_filename(const char *matfile) {
    size_t l = strlen(matfile) + 8;
    char *out = malloc(l);
    if (!out) return NULL;
    snprintf(out, l, "%s.csrbin", matfile);
    return out;
}

static int write_csr_bin(const char *fname, const CSR_info *csr) {
    FILE *f = fopen(fname, "wb");
    if (!f) return -1;
    int64_t nrows = csr->nrows;
    int64_t nz = csr->nzcount;
    if (fwrite(&nrows, sizeof(int64_t), 1, f) != 1) { fclose(f); return -1; }
    if (fwrite(&nz, sizeof(int64_t), 1, f) != 1) { fclose(f); return -1; }
    if (fwrite(csr->row_ptr, sizeof(int), (size_t)(nrows + 1), f) != (size_t)(nrows + 1)) { fclose(f); return -1; }
    if (fwrite(csr->col_idx, sizeof(int), (size_t)nz, f) != (size_t)nz) { fclose(f); return -1; }
    fclose(f);
    return 0;
}

static CSR_info read_csr_bin(const char *fname) {
    CSR_info csr = {0, 0, NULL, NULL};
    FILE *f = fopen(fname, "rb");
    if (!f) return csr;
    int64_t nrows = 0, nz = 0;
    if (fread(&nrows, sizeof(int64_t), 1, f) != 1) { fclose(f); return csr; }
    if (fread(&nz, sizeof(int64_t), 1, f) != 1) { fclose(f); return csr; }
    csr.nrows = (int)nrows;
    csr.nzcount = (int)nz;
    csr.row_ptr = (int *)aligned_alloc_wrapper((size_t)(csr.nrows + 1) * sizeof(int));
    csr.col_idx = (int *)aligned_alloc_wrapper((size_t)csr.nzcount * sizeof(int));
    if (!csr.row_ptr || !csr.col_idx) { fclose(f); clean_csr_vectors(csr); return (CSR_info){0,0,NULL,NULL}; }
    if (fread(csr.row_ptr, sizeof(int), (size_t)(csr.nrows + 1), f) != (size_t)(csr.nrows + 1)) { fclose(f); clean_csr_vectors(csr); return (CSR_info){0,0,NULL,NULL}; }
    if (fread(csr.col_idx, sizeof(int), (size_t)csr.nzcount, f) != (size_t)csr.nzcount) { fclose(f); clean_csr_vectors(csr); return (CSR_info){0,0,NULL,NULL}; }
    fclose(f);
    return csr;
}

CSR_info mat_to_csr(char* file_name) {
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
        free(cache);
    }

    // Open .mat file
    mat_t *matfp = Mat_Open(file_name, MAT_ACC_RDONLY);
    if (!matfp) {
        fprintf(stderr, "MATIO->ERROR: cannot open %s file \n", file_name);
        return (CSR_info){1,1,NULL,NULL};
    }
    
    
    // Read the sparse matrix  with variable name "A" inside struct "Problem" 
    matvar_t *projectvar = Mat_VarRead(matfp, "Problem");
    if (!projectvar) {
        fprintf(stderr, "MATIO->ERROR:'Problem' struct missing in %s\n", file_name);
        Mat_Close(matfp);
        return (CSR_info){1,1, NULL,NULL};
    }

    matvar_t *matvar = Mat_VarGetStructFieldByName(projectvar, "A", 0);
     if (!matvar) {
        fprintf(stderr, "MATIO->ERROR: 'A' matrix missing in struct 'Problem' inside %s\n", file_name);
	      Mat_VarFree(projectvar);
        Mat_Close(matfp);
        return (CSR_info){1,1, NULL,NULL};
    }

    if (matvar->class_type != MAT_C_SPARSE) {
        fprintf(stderr, "MATIO->ERROR: matrix read in %s is not sparse\n", file_name);
	      Mat_VarFree(projectvar);
        Mat_Close(matfp);
        return (CSR_info){1,1,NULL,NULL};
    }

    mat_sparse_t *sparse = (mat_sparse_t* )matvar->data;
    int numrows = matvar->dims[0];   
    int ncols = matvar->dims[1];   
    int *row_ind =(int* ) sparse->ir;  
    int *col_ptr =(int* ) sparse->jc;  

    //Save row pointer and column index data in CSR_info object
    CSR_info csr;

    csr.row_ptr = (int*)aligned_alloc_wrapper((size_t)(numrows+1) * sizeof(int));
    if (csr.row_ptr) memset(csr.row_ptr, 0, (size_t)(numrows+1) * sizeof(int));
    
    // Parallel count nonzeros per row (atomic increments)
    int *row_counts = (int*)calloc((size_t)numrows, sizeof(int));
    if (!row_counts) { Mat_VarFree(projectvar); Mat_Close(matfp); return (CSR_info){1,1,NULL,NULL}; }
    
    cilk_for (int c = 0; c < ncols; ++c) {
        for (int i = col_ptr[c]; i < col_ptr[c+1]; ++i) {
            int r = row_ind[i];
            if (r >= 0 && r < numrows) {
                __atomic_fetch_add(&row_counts[r], 1, __ATOMIC_RELAXED);
            }
        }
    }

    // Build row_ptr via prefix sum (sequential)
    csr.row_ptr[0] = 0;
    for (int i = 0; i < numrows; ++i) {
        csr.row_ptr[i+1] = csr.row_ptr[i] + row_counts[i];
    }

    //Define non-zeros and rows count
    csr.nzcount = csr.row_ptr[numrows];
    csr.nrows = numrows;

    printf("Number of nonzeros: %d\n", csr.nzcount);
  
    //Define col_idx
    csr.col_idx = (int* )aligned_alloc_wrapper(csr.nzcount * sizeof(int));
    
    int *pos = (int*)malloc((size_t)numrows * sizeof(int));
    if (!pos) { free(row_counts); Mat_VarFree(projectvar); Mat_Close(matfp); clean_csr_vectors(csr); return (CSR_info){1,1,NULL,NULL}; }
    for (int i = 0; i < numrows; ++i) pos[i] = csr.row_ptr[i];
    
    // Fill col_idx in parallel; reserve positions via atomic fetch-add
    cilk_for (int c = 0; c < ncols; ++c) {
        for (int j = col_ptr[c]; j < col_ptr[c+1]; ++j) {
            int r = row_ind[j];
            if (r < 0 || r >= numrows) continue;
            int dest = __atomic_fetch_add(&pos[r], 1, __ATOMIC_RELAXED);
            csr.col_idx[dest] = c;
        }
    }

    free(row_counts);
    free(pos);


    // Cleanup dynamically allocated used vectors
    Mat_VarFree(projectvar);
    Mat_Close(matfp);

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

