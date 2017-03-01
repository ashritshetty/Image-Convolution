#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "image.h"

#define TILE_WIDTH 16

__global__ void imgDark(int *outputImage, int *inputImage,
			int width, int height)
{
  int i, j, k, pixel;
	__shared__ int shared_image[TILE_WIDTH+2][TILE_WIDTH+2];
	int kernel2D[3][9] = {-9,-9,-9,-9,72,-9,-9,-9,-9,0,-9,0,-9,45,-9,0,-9,0,1,1,1,1,1,1,1,1,1};
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int kernel[3][3];
  int arg = 0;
	k = 0;
	for(i = 0; i < 3; i++)
	{
		for(j = 0; j < 3; j++)
		{
			kernel[i][j] = kernel2D[arg][k];
			k++;
		}
	}

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
	if(x < width && y < height)
		outputImage[x][y] = pixel/9;
}

int main (int argc, char **argv)
{
	int imageWidth, imageHeight;
	int *hostInputImage, *hostOutputImage;
	int *deviceInputImage, *deviceOutputImage;

	if (argc != 3)
	{
		printf("Usage: ./imageDarken <input-image> <output-image-name>\n");
		exit(1);
	}

	// Read in image and convert to readable format
	read_image_template<int>(argv[1], &hostInputImage, &imageWidth, &imageHeight);

	// Set image size information
	int img_size = imageWidth * imageHeight * sizeof(int);

	hostOutputImage = (int *)malloc(img_size);

	// Allocate memory for image on GPU
	cudaMalloc((void **)&deviceInputImage, img_size);
	cudaMalloc((void **)&deviceOutputImage, img_size);

	// Copy image to device
	cudaMemcpy( deviceInputImage, hostInputImage, img_size, cudaMemcpyHostToDevice );

	// Set kernel dimensions and call kernel
	dim3 dimGrid(ceil(imageWidth / TILE_WIDTH),
		     ceil(imageHeight / TILE_WIDTH));
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	imgDark <<< dimGrid, dimBlock >>>(deviceOutputImage, deviceInputImage, imageWidth, imageHeight);

	// Copy resulting image back to host
	cudaMemcpy( hostOutputImage, deviceOutputImage, img_size, cudaMemcpyDeviceToHost );

	write_image_template<int>(argv[2], hostOutputImage, imageWidth, imageHeight);

	cudaFree(deviceInputImage);
	cudaFree(deviceOutputImage);

	free(hostOutputImage);

	return 0;
}
