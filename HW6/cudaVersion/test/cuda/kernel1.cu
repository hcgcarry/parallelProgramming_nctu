#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

void printImg(int * img,int width,int height);
void copyImg(int *img ,int *h_img,int width ,int height);
__global__ void mandelKernel(float lowerX,float lowerY,int* d_img,int resX,int resY,float stepX,float stepY,int maxIterations) {
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

    d_img[thisY*resX+thisX] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    int imageSize = resX * resY * sizeof(int);
    //int * h_img = (int*) malloc(imageSize);
    int *d_img;
    int BLOCK_SIZE_X = 16;
    int BLOCK_SIZE_Y = 16;
    cudaMalloc((void**)&d_img, resX * resY*sizeof(int));
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 numBlock(resX / BLOCK_SIZE_X, resY / BLOCK_SIZE_Y);
    //cudaMemcpy(d_img,img,resX*resY*sizeof(int),cudaMemcpyHostToDevice);
    mandelKernel<<<numBlock, blockSize>>>(lowerX,lowerY,d_img,resX,resY,stepX,stepY,maxIterations);
    cudaMemcpy(img,d_img,resX*resY*sizeof(int),cudaMemcpyDeviceToHost);
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
