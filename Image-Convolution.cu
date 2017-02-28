#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "Image-Convolution.h"

#define TILE_WIDTH 16
#define BLOCKSIZE 256

__global__ void 1Dconv(int *outputArray, int *inputArray, int kernel[3], int length)
{
	__shared__ int shared_array[BLOCKSIZE+2];
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if((threadIdx.x == 0) && (blockIdx.x == 0))
	{
		shared_array[0] = 0;
	}
	else
	{
		shared_array[0] = inputArray[x-1];
	}
	if((threadIdx.x == blockDim.x-1) && (blockIdx.x == gridDim.x-1))
	{
		shared_array[BLOCKSIZE+1] = 0;
	}
	else
	{
		shared_array[BLOCKSIZE+1] = inputArray[x+1];
	}
	shared_array[x+1] = inputArray[x];
	__syncthreads();
	outputArray[x] = (shared_array[x]*kernel[0]) + (shared_array[x+1]*kernel[1]) + (shared_array[x+2]*kernel[2]);
}

__global__ void 2Dconv(int *outputImage, int *inputImage, int kernel[3][3], int width, int height)
{
	int i, j, pixel;
	__shared__ int shared_image[TILE_WIDTH+2][TILE_WIDTH+2]
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if(blockIdx.x == 0)
		shared_image[0][y+1] = 0;
	else
		shared_image[0][y+1] = inputImage[x-1][y+1];
	if(blockIdx.y == 0)
		shared_image[x+1][0] = 0;
	else
		shared_image[x+1][0] = inputImage[x+1][y-1];
	if(blockIdx.x == gridDim.x-1)
		shared_image[TILE_WIDTH+1][y+1] = 0;
	else
		shared_image[TILE_WIDTH+1][y+1] = inputImage[x+1][y+1];
	if(blockIdx.y == gridDim.y-1)
		shared_image[x+1][TILE_WIDTH+1] = 0;
	else
		shared_image[x+1][TILE_WIDTH+1] = inputImage[x+1][y+1];

	shared_image[x+1][y+1] = inputImage[x][y];
	__syncthreads();

	pixel = 0;
	for(i = -1; i < 2; i++)
	{
		for(j = -1; j < 2; j++)
		{
			pixel = pixel + shared_image[x+i][y+j]*kernel[i+1][j+1];
		}
	}
	outputImage[x][y] = pixel/9;
}

void do_1D_conv(char **argv, int kernel1D[3][3])
{
  int arrayLength, i;
	int *hostInput, *hostOutput;
	int *deviceInput, *deviceOutput;
	int kernel[3]

	for(i = 0; i < 3; i++)
	{
		kernel[i] = kernel1D[atoi(argv[2])][i];
	}

	read_array(argv[3], &hostInput, &arrayLength);
	hostOutput = (int *)calloc(arrayLength, sizeof(int));
	cudaMalloc((void **)&deviceInput, arrayLength * sizeof(int));
	cudaMalloc((void **)&deviceOutput, arrayLength * sizeof(int));
	cudaMemcpy(deviceInput, hostInput, arrayLength * sizeof(int), cudaMemcpyHostToDevice);
	dim3 dimGrid(ceil(arrayLength / BLOCKSIZE, 1, 1);
	dim3 dimBlock((BLOCKSIZE), 1, 1);
	1Dconv <<< dimGrid, dimBlock >>> (deviceOutput, deviceInput, kernel, arrayLength);
	cudaMemcpy(hostOutput, deviceOutput, arrayLength * sizeof(int), cudaMemcpyDeviceToHost);
	write_image_template<int>(argv[4], hostOutput, arrayLength);
	cudaFree(deviceInput);
	cudaFree(deviceOutput);
	free(hostOutput);
}

void do_2D_conv(char **argv, int kernel2D[3][9])
{
	int imageWidth, imageHeight, img_size, i, j, k;
	int *hostInput, *hostOutput;
	int *deviceInput, *deviceOutput;
	int kernel[3][3];

  k = 0;
	for(i = 0; i < 3; i++)
	{
		for(j = 0; j < 3; j++)
		{
			kernel[i][j] = kernel2D[atoi(argv[2])][k];
			k++;
		}
	}

	read_image_template<int>(argv[3], &hostInput, &imageWidth, &imageHeight);
	img_size = imageWidth * imageHeight;
	hostOutput = (int *)calloc(img_size, sizeof(int));
	cudaMalloc((void **)&deviceInput, img_size * sizeof(int));
	cudaMalloc((void **)&deviceOutput, img_size * sizeof(int));
	cudaMemcpy(deviceInput, hostInput, img_size * sizeof(int), cudaMemcpyHostToDevice);
	dim3 dimGrid(ceil(imageWidth / TILE_WIDTH), ceil(imageHeight / TILE_WIDTH));
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	2Dconv <<< dimGrid, dimBlock >>> (deviceOutput, deviceInput, kernel, imageWidth, imageHeight);
	cudaMemcpy(hostOutput, deviceOutput, img_size * sizeof(int), cudaMemcpyDeviceToHost);
	write_image_template<int>(argv[4], hostOutput, imageWidth, imageHeight);
	cudaFree(deviceInput);
	cudaFree(deviceOutput);
	free(hostOutput);
}

int main (int argc, char **argv)
{
	if (argc != 3)
	{
		printf("Usage: ./convolution <dimension> <kernel> <input> <output>\n");
		printf("<dimension> : 0 - 1D\n            : 1 - 2D\n");
		printf("<kernel> : 0 - EDGE\n         : 1 - SHARP\n         : 2 - BLUR\n");
		exit(1);
	}

  int kernel1D[3][3];
	int kernel2D[9][9];

	if(atoi(argv[1]) == 0)
	{
		kernel1D[3][3] = {1,1,1,0,0,0,1,0,1};
		do_1D_conv(argv, kernel1D);
	}
	else
	{
		kernel2D[3][9] = {-9,-9,-9,-9,72,-9,-9,-9,-9,0,-9,0,-9,45,-9,0,-9,0,1,1,1,1,1,1,1,1,1};
		do_2D_conv(argv, kernel2D);
	}

	return 0;
}
