#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include<mpi.h>

#include"matmul.h"

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr,
                        int **a_mat_ptr, int **b_mat_ptr)
{
    scanf("%d %d %d", n_ptr, m_ptr, l_ptr);
    printf("%d %d %d\n",*n_ptr,*m_ptr,*l_ptr);
    int n = *n_ptr, m = *m_ptr, l = *l_ptr;
    *a_mat_ptr = (int *)malloc(n * m * sizeof(int));
    *b_mat_ptr = (int *)malloc(m * l * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            scanf("%d", *a_mat_ptr + i * m + j);
        }
    }
    printf("matrix a\n");
    printMatrix(n,m,*a_mat_ptr);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < l; j++)
        {
            scanf("%d", *b_mat_ptr + i * l + j);
        }
    }
    printf("matrix b\n");
    printMatrix(m,l,*b_mat_ptr);
}

void matrix_multiply(const int n, const int m, const int l,
                     const int *a_mat, const int *b_mat)
{
    int *c_mat = (int*) malloc(sizeof(int) * n *l);
    for(int i=0;i<n;i++){
        for(int j=0;j<l;j++){
            int curPosValue = 0;
            for(int k=0;k<m;k++){
                curPosValue+= a_mat[i*m+k] * b_mat[l*k+j];
            }
            c_mat[i*l+j] = curPosValue;
        }
    }
    
    printf("matrix c\n");
    printMatrix(n,l,c_mat);
}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat){
    free(a_mat);
    a_mat = NULL;
    free(b_mat);
    b_mat = NULL;
}
void printMatrix(int row,int col, int * matrix){
    for(int i =0;i<row;i++){
        for(int j=0;j<col;j++){
            if(j==0) printf("%d",matrix[i*col+j]);
            else printf(" %d",matrix[i*col+j]);
        }
        printf("\n");
    }
}