#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <omp.h>
typedef unsigned uint32_t;
uint32_t next(unsigned*s) ;
void jump(unsigned*s) ;
uint32_t s[4] = {1,2,3,4};
#define  random_max 4294967295

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

    int dest;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int tag = 0; /* tag for messages */
    long long int perProcessTosses = tosses / world_size;
    long long int global_number_in_circle = 0;
    MPI_Status status;

    // TODO: init MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    printf("Hello world from processor %s, rank %d out of %d processors\n",
            processor_name, world_rank, world_size);
    if (world_rank > 0)
    {
        dest = 0;
        // TODO: handle workers
        //jump(s);
        long long int local_number_in_circle = caculateNumInCricle(perProcessTosses,world_rank);
        //printf("local_number_in_circle %l",local_number_in_circle);
        MPI_Send(&local_number_in_circle,1,MPI_LONG, dest, tag, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        //jump(s);
        // TODO: master
        global_number_in_circle = caculateNumInCricle(perProcessTosses,world_rank);
    }

    if (world_rank == 0)
    {
        // TODO: process PI result
        long long int local_number_in_circle ;
        for(int source=1;source<world_size;source++){
           MPI_Recv(&local_number_in_circle, 1,MPI_LONG, source, tag, MPI_COMM_WORLD, &status);
            global_number_in_circle += local_number_in_circle;
        }
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

static inline uint32_t rotl(const uint32_t x, int k) {
	return (x << k) | (x >> (32 - k));
}



uint32_t next(unsigned *s) {
	const uint32_t result = rotl(s[0] + s[3], 7) + s[0];

	const uint32_t t = s[1] << 9;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;

	s[3] = rotl(s[3], 11);

	return result;
}
void jump(unsigned *s) {
	static const uint32_t JUMP[] = { 0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b };

	uint32_t s0 = 0;
	uint32_t s1 = 0;
	uint32_t s2 = 0;
	uint32_t s3 = 0;
	for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
		for(int b = 0; b < 32; b++) {
			if (JUMP[i] & 1 << b) {
				s0 ^= s[0];
				s1 ^= s[1];
				s2 ^= s[2];
				s3 ^= s[3];
			}
			next(s);	
		}
		
	s[0] = s0;
	s[1] = s1;
	s[2] = s2;
	s[3] = s3;
}

