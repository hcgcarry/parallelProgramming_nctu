#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

void printImg(int * img,int width,int height);
void copyImg(int *img ,int *h_img,int width ,int height);
__global__ void mandelKernel(float lowerX,float lowerY,int* d_img,int resX,int resY,float stepX,float stepY,int maxIterations,size_t pitchSize) {
    // To avoid error caused by the floating number, use the following pseudo code

    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    float x = lowerX + thisX * stepX;
    float y = lowerY + thisY * stepY;
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

    d_img[thisY*pitchSize+thisX] = i;
    //d_img[thisY*pitchSize+thisX] = 255;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
	///resX is img width , resY is img height
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    int imageSize = resX * resY * sizeof(int);
    // host mem
    int * h_img ;
    cudaHostAlloc((void**)&h_img,imageSize,cudaHostAllocMapped);
    // device mem
    int *d_img;
    size_t pitchSize;
    cudaMallocPitch((void**)&d_img,&pitchSize,resX*sizeof(int),resY); 
    //launch kernel
    int BLOCK_SIZE_X = 32;
    int BLOCK_SIZE_Y = 16;
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 numBlock(resX / BLOCK_SIZE_X, resY / BLOCK_SIZE_Y);
    //cudaMemcpy(d_img,img,resX*resY*sizeof(int),cudaMemcpyHostToDevice);
    //note that pitchSize should be diveded by sizeof(int) because pitchSize
    mandelKernel<<<numBlock, blockSize>>>(lowerX,lowerY,d_img,resX,resY,stepX,stepY,maxIterations,pitchSize/sizeof(int));
    cudaDeviceSynchronize();
    //cudaMemcpy(h_img,d_img,resX*resY*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy2D(h_img,resX*sizeof(int),d_img,pitchSize,resX*sizeof(int),resY, cudaMemcpyDeviceToHost);
    copyImg(img,h_img,resX,resY);
    //printf("width %d height %d \n",resX,resY);
    //printImg(img,resX,1);
    cudaFree(d_img);
    return ;
}
void copyImg(int *img ,int *h_img,int width ,int height){
		for(int j=0;j<height;j++){
	for(int i =0 ;i<width;i++){
	 	img[j*width + i ]  = h_img[j*width+i];
	}
		}
			
}

void printImg(int * img,int width,int height){
		for(int j=0;j<height;j++){
	for(int i=0;i<width;i++){
			printf("%d ",img[j*height+i]);
	}
	printf("\n");
		}
}
