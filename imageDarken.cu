#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "image.h"

#define TILE_WIDTH 16

#define emax(x, y) ((x) >= (y)) ? (x) : (y)
#define emin(x, y) ((x) <= (y)) ? (x) : (y)
__global__ void imgDark(int *outputImage, int *inputImage,
			int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height)
	{
		int offset = y * width + x;
		outputImage[offset] = emax(inputImage[offset] - 75, 0);
	}
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
