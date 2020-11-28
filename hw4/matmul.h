
void printMatrix(int row,int col, int * matrix);
void destruct_matrices(int *a_mat, int *b_mat);

void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat);
void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr);
void caculatePartialMatrix(int *a_matrix, int *b_matrix, int *c_matrix, int n, int m, int l);