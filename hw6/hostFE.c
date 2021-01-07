#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"
//cl_program load_program(cl_context context, const char* filename,cl_device_id);

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth * sizeof(int);
	int imageSize = imageHeight * imageWidth * sizeof(int);

	//function parameter
	// OpenCL
	//int global_work_size[2] = {imageWidth,imageHeight};
	//printf("%d %d\n" ,imageHeight,imageWidth);
	size_t global_work_size[2] = {imageWidth,imageHeight};
	size_t local_work_size[2] = {8,16};
	cl_kernel  convolution;
	cl_command_queue command_queue = NULL;
	cl_int ret;
	cl_int err;

	command_queue = clCreateCommandQueue(*context, *device, 0, &ret);
	if (ret != CL_SUCCESS)
	 {
	   printf("Failed to create command queue %d\n", (int) ret);
		goto error;
	       return 0;
	}

	//program = load_program(*context, "kernel.cl",*device);

	// create kernel objects from compiled program	
	convolution = clCreateKernel(*program, "convolution", NULL);
	if(convolution== 0) {
		printf("Error, can't load kernel convolution\n");
        //goto error;
    }

	// Memory Buffer
	cl_mem filter_cl = clCreateBuffer(*context, CL_MEM_READ_ONLY,filterSize , NULL, &err);
	if(err != CL_SUCCESS){
		printf("create buffer error \n");
	}
	cl_mem inputImage_cl= clCreateBuffer(*context, CL_MEM_READ_ONLY,imageSize, NULL, &err);
	if(err != CL_SUCCESS){
		printf("create buffer error \n");
	}
	cl_mem outputImage_cl = clCreateBuffer(*context, CL_MEM_WRITE_ONLY,imageSize, NULL, &err);
	if(err != CL_SUCCESS){
		printf("create buffer error \n");
	}



    // copy data to cl buffer
    cl_event startEvt;
    err = clEnqueueWriteBuffer(command_queue, filter_cl, CL_TRUE, 0, filterSize, filter , 0, NULL, NULL);
    	if(err != CL_SUCCESS){

		printf("enqueuewritebuffer error %d\n",err);
	}
	err = clEnqueueWriteBuffer(command_queue, inputImage_cl , CL_TRUE, 0,imageSize , inputImage, 0, NULL, &startEvt);
    clWaitForEvents(1, &startEvt);
    	if(err != CL_SUCCESS){

		printf("enqueuewritebuffer error %d\n",err);
	}

	/// set kernel args
	err = clSetKernelArg(convolution, 0, sizeof(cl_int), &filterWidth);
	err |= clSetKernelArg(convolution, 1, sizeof(cl_mem), &filter_cl);
	err |= clSetKernelArg(convolution, 2, sizeof(cl_int), &imageHeight);
	err |= clSetKernelArg(convolution, 3, sizeof(cl_int), &imageWidth);
	err |= clSetKernelArg(convolution, 4, sizeof(cl_mem), &inputImage_cl);
	err |= clSetKernelArg(convolution, 5, sizeof(cl_mem), &outputImage_cl);
    	if(err != CL_SUCCESS){
		printf("cl set kernel args error \n");
	}

        err = clEnqueueNDRangeKernel(command_queue, convolution, 2, NULL,&global_work_size,&local_work_size, 0, NULL, NULL);

        if(err == CL_SUCCESS){}
        else
        {
            printf("Error: can't enqueue kernel %d\n",err);
        }


    cl_event finalEvt;
    err =  clEnqueueReadBuffer(command_queue, outputImage_cl, CL_TRUE, 0, imageSize, outputImage, 0, NULL, &finalEvt);
    	if(err != CL_SUCCESS){
		printf("enqueue read buffer error %d\n",err);
	}
    clWaitForEvents(1, &finalEvt);
return 0;
error:
	clFlush(command_queue);
	clFinish(command_queue);
	clReleaseKernel(convolution);
	clReleaseProgram(*program);

    clReleaseMemObject(filter_cl);
    clReleaseMemObject(outputImage_cl);
    clReleaseMemObject(inputImage_cl);

    clReleaseCommandQueue(command_queue);
    clReleaseContext(*context);

    /* free host resources */
	return 0;

}

