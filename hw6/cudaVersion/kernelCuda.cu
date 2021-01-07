__global__ void convolution(int filterWidth,float *filter,int imageHeight,int imageWidth,float *inputImage,float *outputImage)
{

	int i= blockIdx.x * blockDim.x + threadIdx.x;
	int j= blockIdx.y * blockDim.y + threadIdx.y;
	// Iterate over the rows of the source image
	int halffilterSize = filterWidth >> 1 ;
	float sum;
	int  k, l;
            sum = 0; // Reset sum for new source pixel
            // Apply the filter to the neighborhood
            for (k = -halffilterSize; k <= halffilterSize; k++)
            {
                for (l = -halffilterSize; l <= halffilterSize; l++)
                {
                    if (j + k >= 0 && j + k < imageHeight &&
                        i + l >= 0 && i + l < imageWidth)
                    {
                        sum += inputImage[(j + k) * imageWidth + i + l] *
                               filter[(k + halffilterSize) * filterWidth +
                                      l + halffilterSize];
                    }
                }
            }
            outputImage[j * imageWidth + i] = sum;
            //outputImage[j * imageWidth + i] = 255;
}
extern "C" void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
		            float *inputImage, float *outputImage)
{
	
	float * d_filter,*d_inputImage,*d_outputImage;
	int filterSize  = filterWidth * filterWidth * sizeof(float);
	int inputImageSize = imageHeight * imageWidth * sizeof(int);
	int outputImageSize = inputImageSize;
	cudaMalloc((void**)&d_filter,filterSize);
	cudaMalloc((void**)&d_inputImage,inputImageSize);
	cudaMalloc((void**)&d_outputImage,outputImageSize);
	cudaMemcpy(d_filter,filter,filterSize,cudaMemcpyHostToDevice);
	cudaMemcpy(d_inputImage,inputImage,inputImageSize,cudaMemcpyHostToDevice);
	int block_size_x = 16;
	int block_size_y = 16;
	dim3 blockSize(block_size_x,block_size_y);
	dim3 numBlock(imageWidth/block_size_x,imageHeight/block_size_y);
	convolution<<<numBlock,blockSize>>>(filterWidth,d_filter,imageHeight,imageWidth,d_inputImage,d_outputImage);
	cudaMemcpy(outputImage,d_outputImage,outputImageSize,cudaMemcpyDeviceToHost);
	cudaFree(d_outputImage);
	cudaFree(d_inputImage);
	cudaFree(d_filter);
}

