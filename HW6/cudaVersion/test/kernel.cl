//__kernel void convolution(){ }
__kernel void convolution(const int filterWidth,__global float *filter,const int imageHeight,const int imageWidth, __global float *inputImage,__global float *outputImage)
{

	int i=  get_global_id(0);
	int j= get_global_id(1);
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
