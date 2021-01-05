//__kernel void convolution(__constant int filterWidth,__constant float *filter,__constant int imageHeight,__constant int imageWidth, __constant float *inputImage,__global float *outputImage,__local int* localInputImage)
__kernel void convolution(__constant int filterWidth,__constant float *filter,__constant int imageHeight,__constant int imageWidth, __constant float *inputImage,__global float *outputImage)
{

	int i=  get_global_id(0);
	int j= get_global_id(1);
	int localI = get_local_id(0);
	int localJ = get_local_id(1);
	int baseImgOffset = j  * imageWidth + i;
	// copy inputImage from global memory to local memory;
/*
	if(localI == 0 || localJ == 0 || localI == get_local_size(0)-1 || localJ == get_local_size(1) ){
		
	}
	else{
		localImage = localInputImage[baseImgOffset];
	}
*/
	
	// Iterate over the rows of the source image
	int halffilterSize = filterWidth >> 1 ;
	int filterIndex =0;
	float sum = 0.0;
            // Apply the filter to the neighborhood
            for (int k = -halffilterSize; k <= halffilterSize; k++)
            {
	    	int curImageOffset = baseImgOffset + k*imageWidth;
                for (int l = -halffilterSize; l <= halffilterSize; l++,filterIndex++)
                {
                    if (j+k>= 0 && j+k< imageHeight &&
                        i+l>= 0 && i+l< imageWidth)
                    {
                        sum += inputImage[curImageOffset + l] *
                               filter[ filterIndex];
                    }
                }
            }
            outputImage[baseImgOffset] = sum;
            //outputImage[j * imageWidth + i] = 255;
}
