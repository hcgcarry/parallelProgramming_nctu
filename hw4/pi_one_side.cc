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
    // Create the window
    long long int window_buffer = 0;
    MPI_Win window;
    MPI_Win_create(&window_buffer, sizeof(long long int), sizeof(long long int), MPI_INFO_NULL, MPI_COMM_WORLD, &window);

    if (world_rank == 0)
    {
        // Master
    }
    MPI_Win_fence(0, window);
    if(world_rank >0) 
    {
        MPI_Accumulate(&local_number_in_circle, 1, MPI_LONG, 0, 0, 1, MPI_LONG, MPI_SUM, window);
        // Workers
    }
    MPI_Win_fence(0, window);


    if (world_rank == 0)
    {
        global_number_in_circle = local_number_in_circle;
        global_number_in_circle += window_buffer;
        // TODO: handle PI result
        pi_result = (double)4*global_number_in_circle / tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    MPI_Win_free(&window);
    
    MPI_Finalize();
    return 0;
}