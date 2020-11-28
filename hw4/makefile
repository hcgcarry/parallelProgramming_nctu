
all:main.cc matmul.c
	mpicxx main.cc matmul.c -o matmul; 
run:	
	mpirun -np 4 --map-by ppr:1:node  --hostfile hosts matmul < matdata



block_linear:
	mpicxx pi_block_linear.cc -o pi_block_linear; mpirun  --map-by ppr:1:node -np 4 --hostfile hosts pi_block_linear 1000000000


block_tree:
	mpicxx pi_block_tree.cc -o pi_block_tree; mpirun  --map-by ppr:1:node -np 4 --hostfile hosts pi_block_tree 1000000000

nonblock_linear:
	mpicxx pi_nonblock_linear.cc -o pi_nonblock_linear; mpirun -np 4 --hostfile hosts pi_nonblock_linear 1000000000

gather:
	mpicxx pi_gather.cc -o pi_gather; mpirun -np 4 --hostfile hosts pi_gather 1000000000

reduce:
	mpicxx pi_reduce.cc -o pi_reduce; mpirun -np 4 --hostfile hosts pi_reduce 1000000000

one_side:
	mpicxx pi_one_side.cc -o pi_one_side; mpirun -np 4 --hostfile hosts pi_one_side 1000000000
ping_pong:
	mpicxx ping_pong.c -o ping_pong; mpirun -np 2 --hostfile hosts ping_pong 1000000000

clean:
	rm matmul
