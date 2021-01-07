default: conv

CC = gcc-10
FLAGS = -O3 -lOpenCL -m64 -w -g
CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC'
CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3


cudaOBJS = mainCuda.o bmpfuncs.o kernelCuda.o serialConv.o helper.o

convCuda: $(cudaOBJS)
	nvcc -o $@ $^ $(CUDA_LINK_FLAGS) $(FLAGS)
kernelCuda.o:kernelCuda.cu
	nvcc ${CUDA_COMPILE_FLAGS} -c kernelCuda.cu -o $@

%.o: %.c
	$(CC) -c $(FLAGS) $< -o $@ 


clean:
	rm -f convCuda *.o output.bmp ref.bmp
