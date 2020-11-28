#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <omp.h>

#define random_max 4294967295
long long int caculateNumInCricle(long long int perProcessTosses, int world_rank)
{
    long long int local_number_in_circle = 0;
    float x, y, distance_squared;
    unsigned int seedx = 123 * world_rank;
    unsigned int seedy = seedx + 1;
    #pragma omp parallel for reduction(+:local_number_in_circle)
    for (long long toss = 0; toss < perProcessTosses; toss++)
    {

        //printf("world_rank %d, toss %d\n",world_rank,toss);
        /*
            x = (double)next(s) * 2 / random_max - 1;
            y = (double)next(s) * 2 / random_max - 1;
            */
        x = (double)rand_r(&seedx) * 2 / RAND_MAX - 1;
        y = (double)rand_r(&seedy) * 2 / RAND_MAX - 1;
        distance_squared = x * x + y * y;

        if (distance_squared <= 1)
            local_number_in_circle++;
    }
    return local_number_in_circle;
}

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---
    int dest = 0;
    int tag = 0; /* tag for messages */
    long long int global_number_in_circle = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    long long int perProcessTosses = tosses / world_size;
    MPI_Status status;

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    printf("Hello world from processor %s, rank %d out of %d processors\n",
           processor_name, world_rank, world_size);

    // Print off a hello world message
    // TODO: binary tree redunction
    // caculate

    long long int local_number_in_circle = caculateNumInCricle(perProcessTosses, world_rank);

    //fflush(stdout);

    // reduction
    int curIterStepSize = 1;
    while (curIterStepSize < world_size)
    {
        //printf("0cur world_rank %d ,curIterStepSize:%d\n",world_rank ,curIterStepSize);
        //fflush(stdout);
        if (world_rank % curIterStepSize == 0)
        {
        //printf("1cur world_rank %d ,curIterStepSize:%d\n",world_rank ,curIterStepSize);
            if ((world_rank /curIterStepSize)% 2 == 0)
            {
                // notice this type ,if let it be int ,when we call receive it will clear another four byte address which
                // may be some variable
                long long int receive_number_in_circles;
                int source;
                source = world_rank + curIterStepSize;
                //printf("curRank %d, recvSource %d\n", world_rank, source);
                //fflush(stdout);
                MPI_Recv(&receive_number_in_circles, 1, MPI_LONG, source, tag, MPI_COMM_WORLD, &status);
                local_number_in_circle += receive_number_in_circles;
            }
            else
            {
                int dest = world_rank - curIterStepSize;
                //printf("curRank %d, sendDest %d\n", world_rank, dest);
                //fflush(stdout);
                MPI_Send(&local_number_in_circle, 1, MPI_LONG, dest, tag, MPI_COMM_WORLD);
            }
        }
        //printf("2 cur world_rank %d ,curIterStepSize:%d\n",world_rank ,curIterStepSize);
        //MPI_Barrier(MPI_COMM_WORLD);
        //printf("3 cur world_rank %d ,curIterStepSize:%d\n",world_rank ,curIterStepSize);
        curIterStepSize  = curIterStepSize * 2;
    }
    //printf("after while\n");
    //fflush(stdout);

    if (world_rank == 0)
    {
        // TODO: PI result
        global_number_in_circle = local_number_in_circle;
        pi_result = (double)4 * global_number_in_circle / tosses;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
