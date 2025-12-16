#ifndef BFS_H
#define BFS_H

/* Common header for BFS/CC algorithms (used by BFScilk.c) */

int coloring_cc(int n, const long long *row_ptr, const int *col_idx,
                int *labels, int max_iters);
int count_components(int n, const int *labels);
void print_sample_labels(int n, const int *labels, int max_print);

#endif /* BFS_H */
