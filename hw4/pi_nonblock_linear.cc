#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

long long int caculateNumInCricle(long long int perProcessTosses,int world_rank){
        long long int local_number_in_circle= 0;
        float x,y,distance_squared;
        unsigned int seedx = 123 * world_rank;
        unsigned int seedy = seedx +1;
        #pragma omp parallel for reduction(+:local_number_in_circle)
        for (long long int toss = 0; toss < perProcessTosses; toss++)
        {
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
    // TODO: MPI init
    int dest=0;
    int tag = 0; /* tag for messages */
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    long long int perProcessTosses = tosses / world_size;

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    /*
    printf("Hello world from processor %s, rank %d out of %d processors\n",
            processor_name, world_rank, world_size);
            */

    long long int global_number_in_circle = 0;

    long long int local_number_in_circle = caculateNumInCricle(perProcessTosses,world_rank);
    if (world_rank > 0)
    {
        //printf("local_number_in_circle %l",local_number_in_circle);
        MPI_Send(&local_number_in_circle,1,MPI_LONG, dest, tag, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        MPI_Request requests[world_size-1];
        long long int receive_local_number_in_circle[world_size-1];
        for(int i=1;i<world_size;i++){
            MPI_Irecv(&receive_local_number_in_circle[i-1], 1, MPI_LONG, i, tag, MPI_COMM_WORLD, &(requests[i-1]));
        }
        global_number_in_circle = local_number_in_circle ;
        MPI_Waitall(world_size-1,requests,MPI_STATUS_IGNORE);
        for(int i=1;i<world_size;i++){
            global_number_in_circle+=receive_local_number_in_circle[i-1];
        }
    }

    if (world_rank == 0)
    {
        // TODO: PI result

        pi_result = (double)4*global_number_in_circle / tosses;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
