#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "matmul.h"
#define MASTER 0      /* taskid of first task */
#define FROM_MASTER 1 /* setting a message type */
#define FROM_WORKER 2 /* setting a message type */

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr,
                        int **a_mat_ptr, int **b_mat_ptr)
{
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    #ifdef debug
    printf("Hello world from processor %s, rank %d out of %d processors\n",
           processor_name, world_rank, world_size);
    #endif

    /// master
    if (world_rank == MASTER)
    {

        scanf("%d %d %d", n_ptr, m_ptr, l_ptr);
        #ifdef debug
        printf("%d %d %d\n", *n_ptr, *m_ptr, *l_ptr);
        #endif
        int n = *n_ptr, m = *m_ptr, l = *l_ptr;
        *a_mat_ptr = (int *)malloc(n * m * sizeof(int));
        *b_mat_ptr = (int *)malloc(m * l * sizeof(int));
        // construct matrix a
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                scanf("%d", *a_mat_ptr + i * m + j);
            }
        }
        // construct matrix b
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < l; j++)
            {
                scanf("%d", *b_mat_ptr + i * l + j);
            }
        }
        /// print matrix
        #ifdef debug
        printf("matrix a\n");
        printMatrix(n, m, *a_mat_ptr);
        printf("matrix b\n");
        printMatrix(m, l, *b_mat_ptr);
        #endif
    }
    //printf("-1 hello rank %d\n", world_rank);
}

void matrix_multiply(const int n, const int m, const int l,
                     const int *a_mat, const int *b_mat)
{
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    /* Send matrix data to the worker tasks */
    int *global_c_matrix;
    //printf("0 hello rank %d\n", world_rank);
    if (world_rank == MASTER)
    {
        int messageType = FROM_MASTER;
        for (int dest = 1; dest < world_size; dest++)
        {
            MPI_Send(&n, 1, MPI_INT, dest, messageType, MPI_COMM_WORLD);
        }
        global_c_matrix = (int *)malloc(sizeof(int) * n * l);

        int numOfWrokers = (n < world_size) ? n : world_size;
        int rowPerWorker = n / numOfWrokers;
        int extra = n % numOfWrokers;
        int offset = 0;
        // master workload
        int masterWorkerHandleRows = (extra) ? rowPerWorker + 1 : rowPerWorker;
        offset += masterWorkerHandleRows;
        int curworkerHandleRows ;
        // other worker workload
        for (int dest = 1; dest < numOfWrokers; dest++)
        {

            int workerHandleRows = (dest < extra) ? rowPerWorker + 1 : rowPerWorker;
            //printf("Sending %d rows to task %d offset=%d\n", workerHandleRows, dest, offset);
            MPI_Send(&workerHandleRows, 1, MPI_INT, dest, messageType, MPI_COMM_WORLD);
            MPI_Send(&m, 1, MPI_INT, dest, messageType, MPI_COMM_WORLD);
            MPI_Send(&l, 1, MPI_INT, dest, messageType, MPI_COMM_WORLD);
            MPI_Send(&offset, 1, MPI_INT, dest, messageType, MPI_COMM_WORLD);
            MPI_Send(a_mat + offset * m, workerHandleRows * m, MPI_INT, dest, messageType, MPI_COMM_WORLD);
            MPI_Send(b_mat, m * l, MPI_INT, dest, messageType, MPI_COMM_WORLD);
            offset = offset + workerHandleRows;
        }
        ///////////// caculate matrix
        printf("-----------rank %d ,workerHandleRows %d, m %d, l %d------------\n", world_rank,masterWorkerHandleRows,m,l);
        caculatePartialMatrix((int *)a_mat, (int *)b_mat, (int *)global_c_matrix, masterWorkerHandleRows, m, l);
        /* Receive results from worker tasks */
        MPI_Status status;
        messageType = FROM_WORKER;
        for (int source = 1; source < numOfWrokers; source++)
        {
            int workerHandleRows;
            MPI_Recv(&offset, 1, MPI_INT, source, messageType, MPI_COMM_WORLD, &status);
            MPI_Recv(&workerHandleRows, 1, MPI_INT, source, messageType, MPI_COMM_WORLD, &status);
            MPI_Recv(global_c_matrix + offset * l, workerHandleRows * l, MPI_INT, source, messageType, MPI_COMM_WORLD, &status);
            //printf("Received results from task %d\n", source);
        }
        //printf("matrix C \n");
        printMatrix(n, l, global_c_matrix);
        free(global_c_matrix);
    }
    //**************************** worker task ************************************/
    else if (world_rank > MASTER)
    {
        int n;
        int messageType = FROM_MASTER;
        MPI_Status status;
        MPI_Recv(&n, 1, MPI_INT, MASTER, messageType, MPI_COMM_WORLD, &status);
        if (world_rank < n)
        {
            //printf("n %d ,1 hello rank %d %d\n", n, world_rank, world_rank < n);
            //receive matrix info
            int offset;
            int workerHandleRows;
            int m, l;
            MPI_Recv(&workerHandleRows, 1, MPI_INT, MASTER, messageType, MPI_COMM_WORLD, &status);
            MPI_Recv(&m, 1, MPI_INT, MASTER, messageType, MPI_COMM_WORLD, &status);
            MPI_Recv(&l, 1, MPI_INT, MASTER, messageType, MPI_COMM_WORLD, &status);
            MPI_Recv(&offset, 1, MPI_INT, MASTER, messageType, MPI_COMM_WORLD, &status);
            int *matrix_a = (int *)malloc(sizeof(int) * workerHandleRows * m);
            int *matrix_b = (int *)malloc(sizeof(int) * m * l);
            int *matrix_c = (int *)malloc(sizeof(int) * workerHandleRows * l);
            MPI_Recv(matrix_a, workerHandleRows * m, MPI_INT, MASTER, messageType, MPI_COMM_WORLD, &status);
            MPI_Recv(matrix_b, m * l, MPI_INT, MASTER, messageType, MPI_COMM_WORLD, &status);
            #ifdef debug
            printf("-----------rank %d , receive ------------\n", world_rank);
            printf("workerHandleRows %d ,m %d,l %d,offset %d\n", workerHandleRows, m, l, offset);
            #endif
            /////////// caculate matrix
            printf("-----------rank %d ,workerHandleRows %d, m %d, l %d------------\n", world_rank,workerHandleRows,m,l);
            caculatePartialMatrix(matrix_a, matrix_b, matrix_c, workerHandleRows, m, l);
            #ifdef debug
            printf("-------rank %d result:----------\n", world_rank);
            #endif
            //printMatrix(workerHandleRows, l, matrix_c);
            /// send result to master
            messageType = FROM_WORKER;
            MPI_Send(&offset, 1, MPI_INT, MASTER, messageType, MPI_COMM_WORLD);
            MPI_Send(&workerHandleRows, 1, MPI_INT, MASTER, messageType, MPI_COMM_WORLD);
            MPI_Send(matrix_c, workerHandleRows * l, MPI_INT, MASTER, messageType, MPI_COMM_WORLD);
            free(matrix_a);
            free(matrix_b);
            free(matrix_c);
        }
    }
    //printf("2 hello rank %d\n", world_rank);
}

void caculatePartialMatrix(int *a_matrix, int *b_matrix, int *c_matrix, int n, int m, int l)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < l; j++)
        {
            int curPosValue = 0;
            for (int k = 0; k < m; k++)
            {
                curPosValue += a_matrix[i * m + k] * b_matrix[l * k + j];
            }
            c_matrix[i * l + j] = curPosValue;
        }
    }
}
// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat)
{
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    //printf("destruct_matrices rank%d\n", world_rank);

    if (world_rank == MASTER)
    {
        //printf("in free\n");
        free(a_mat);
        a_mat = NULL;
        free(b_mat);
        b_mat = NULL;
    }
}
void printMatrix(int row, int col, int *matrix)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            if (j == 0)
                printf("%d", matrix[i * col + j]);
            else
                printf(" %d", matrix[i * col + j]);
        }
        printf("\n");
    }
}
