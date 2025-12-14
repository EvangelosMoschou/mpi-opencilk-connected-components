#ifndef MAT_TO_CSR_H
#define MAT_TO_CSR_H
#include <stdlib.h>

typedef struct {
  long long nzcount;
  int nrows;

  long long *row_ptr;
  int *col_idx;
} CSR_info;

void clean_csr_vectors(CSR_info vecs);
// Compute edge-balanced partition
int *compute_balanced_partition(const char *fname, int size, int *out_n);
// Read partial CSR for distributed loading
CSR_info read_csr_bin_part(const char *fname, int start_row, int end_row,
                           int *out_global_n);
CSR_info mat_to_csr(char *file_name);
void *aligned_alloc_wrapper(size_t size);

#endif
