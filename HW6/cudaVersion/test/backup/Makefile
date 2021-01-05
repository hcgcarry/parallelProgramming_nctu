default: conv

CC = gcc-10
FLAGS = -O3 -lOpenCL -m64 -ffloat-store -w -g
CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC'
CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3


OBJS = main.o bmpfuncs.o hostFE.o serialConv.o helper.o
cudaOBJS = mainCuda.o bmpfuncs.o kernelCuda.o serialConv.o helper.o

conv: $(OBJS)
	$(CC) -o $@ $(OBJS) $(FLAGS)
convCuda: $(cudaOBJS)
	nvcc -o $@ $(OBJS) $(CUDA_COMPILE_FLAGS)
kernelCuda.o:kernelCuda.cu
	nvcc ${CUDA_COMPILE_FLAGS} -c kernelCuda.cu -o $@

%.o: %.c
	$(CC) -c $(FLAGS) $< -o $@


clean:
	rm -f conv *.o output.bmp ref.bmp
