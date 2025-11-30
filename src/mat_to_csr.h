#ifndef MAT_TO_CSR_H
#define MAT_TO_CSR_H

typedef struct{
   int nzcount;
   int nrows;

   int *row_ptr;
   int *col_idx;
}CSR_info;

void clean_csr_vectors(CSR_info vecs);
CSR_info mat_to_csr(char* file_name);
void* aligned_alloc_wrapper(size_t size);

#endif
