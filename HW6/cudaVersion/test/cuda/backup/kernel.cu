#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

void printImg(int * img,int width,int height);
void copyImg(int *img ,int *h_img,int size);
__global__ void mandelKernel(float lowerX,float lowerY,int* d_img,int resX,int resY,float stepX,float stepY,int maxIterations,int row_offset) {
    // To avoid error caused by the floating number, use the following pseudo code
    //int thisX = blockIdx.x * blockDim.x + threadIdx.x + row_offset;
    //int thisX = blockIdx.x * blockDim.x + threadIdx.x + row_offset;
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    float x = lowerX + thisX * stepX;
    float y = lowerY + (thisY+row_offset) * stepY ;
    float c_re = x,c_im = y;
    float z_re = x,z_im= y;
    int i,count=0;
    for (i = 0; i < maxIterations; ++i)
    {
        if (z_re * z_re + z_im * z_im > 4.f)
		break;
	
	//count ++;
        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    d_img[thisY*resX+thisX] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{

	cudaStream_t stream1,stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    int imageSize = resX * resY * sizeof(int);
    int * h_img_1,*h_img_2 ;
    cudaHostAlloc(&h_img_1,imageSize/2,cudaHostAllocMapped);
    cudaHostAlloc(&h_img_2,imageSize/2,cudaHostAllocMapped);
    int *d_img_1,*d_img_2;
    cudaMalloc((void**)&d_img_1, imageSize/2);
    cudaMalloc((void**)&d_img_2, imageSize/2);

    int BLOCK_SIZE_X = 20;
    int BLOCK_SIZE_Y = 8;
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 numBlock(resX / BLOCK_SIZE_X, resY / BLOCK_SIZE_Y/2);
    //cudaMemcpy(d_img,img,resX*resY*sizeof(int),cudaMemcpyHostToDevice);
    int row_offset_1=0,row_offset_2 = resY/2;
    mandelKernel<<<numBlock, blockSize,0,stream1>>>(lowerX,lowerY,d_img_1,resX,resY,stepX,stepY,maxIterations,row_offset_1);
    mandelKernel<<<numBlock, blockSize,0,stream2>>>(lowerX,lowerY,d_img_2,resX,resY,stepX,stepY,maxIterations,row_offset_2);
    cudaMemcpyAsync(h_img_1,d_img_1,imageSize/2,cudaMemcpyDeviceToHost,stream1);
    //cudaMemcpyAsync(h_img+resX*resY,d_img_2,imageSize/2,cudaMemcpyDeviceToHost,stream2);
    cudaMemcpyAsync(h_img_2,d_img_2,imageSize/2,cudaMemcpyDeviceToHost,stream2);
    cudaDeviceSynchronize();
    copyImg(img,h_img_1,imageSize/2);
    //printImg(h_img_2,resX,resY/2);
    copyImg((img + resX*resY/2),h_img_2,imageSize/2);
    //printf("width %d height %d \n",resX,resY);
    //printImg(img,resX,1);
    return ;
}
void copyImg(int *img ,int *h_img,int size){
	memcpy(img,h_img,size);
			
}

void printImg(int * img,int width,int height){
		for(int j=0;j<height;j++){
	for(int i=0;i<width;i++){
			printf("%d ",img[j*height+i]);
	}
	printf("\n");
		}
}
