#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "utilities.h"

#define TILE_WIDTH 16
#define SH_MEM_WIDTH 18

__global__ void conv2d(int *outputImage, int *inputImage, int width, int height, int arg)
{
  int i, j, k, pixel;
  __shared__ int shared_image[SH_MEM_WIDTH][SH_MEM_WIDTH];
  int kernel2D[3][9] = {-9,-9,-9,-9,72,-9,-9,-9,-9,9,-9,-9,-9,81,-9,-9,-9,-9,-9,-9,0,-9,0,9,0,9,9};
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int kernel[3][3];

  k = 0;
  for(i = 0; i < 3; i++)
  {
    for(j = 0; j < 3; j++)
    {
      kernel[i][j] = kernel2D[arg][k];
      k++;
    }
  }

  shared_image[(x%TILE_WIDTH)+1][(y%TILE_WIDTH)+1] = inputImage[x*width+y];
  __syncthreads();

  if(x%TILE_WIDTH == 0)
    shared_image[0][(y%TILE_WIDTH+1)] = shared_image[1][(y%TILE_WIDTH)+1];

  if(x%TILE_WIDTH == TILE_WIDTH-1)
    shared_image[SH_MEM_WIDTH-1][(y%TILE_WIDTH)+1] = shared_image[SH_MEM_WIDTH-2][(y%TILE_WIDTH)+1];

  if(y%TILE_WIDTH == 0)
    shared_image[(x%TILE_WIDTH)+1][0] = shared_image[(x%TILE_WIDTH)+1][1];

  if(y%TILE_WIDTH == TILE_WIDTH-1)
    shared_image[(x%TILE_WIDTH)+1][SH_MEM_WIDTH-1] = shared_image[(x%TILE_WIDTH)+1][SH_MEM_WIDTH-2];
  __syncthreads();

  if((x%TILE_WIDTH)+(y%TILE_WIDTH) == 0)
  {
    shared_image[0][0] = shared_image[1][1];
    shared_image[SH_MEM_WIDTH-1][SH_MEM_WIDTH-1] = shared_image[SH_MEM_WIDTH-2][SH_MEM_WIDTH-2];
    shared_image[0][SH_MEM_WIDTH-1] = shared_image[1][SH_MEM_WIDTH-2];
    shared_image[SH_MEM_WIDTH-1][0] = shared_image[SH_MEM_WIDTH-2][1];
  }
  __syncthreads();

  pixel = 0;
  for(i = 0; i < 3; i++)
  {
    for(j = 0; j < 3; j++)
    {
      pixel = pixel + shared_image[(x%TILE_WIDTH)+i][(y%TILE_WIDTH)+j]*kernel[i][j];
    }
  }
  if(x < width && y < height)
  {
    outputImage[x*width+y] = pixel/9;
  }
}

int main (int argc, char **argv)
{
  int imageWidth, imageHeight, img_size, arg;
  int *hostInputImage, *hostOutputImage;
  int *deviceInputImage, *deviceOutputImage;

  if (argc != 4)
  {
    printf("Usage    : ./convolution_2d <input> <output> <kernel>\n");
    printf("<kernel> : 0 - EDGE\n         : 1 - SHARP\n         : 2 - EMBOSS\n");
    exit(1);
  }

  arg = atoi(argv[3]);
  read_image_template<int>(argv[1], &hostInputImage, &imageWidth, &imageHeight);
  img_size = imageWidth * imageHeight * sizeof(int);
  hostOutputImage = (int *)malloc(img_size);
  cudaMalloc((void **)&deviceInputImage, img_size);
  cudaMalloc((void **)&deviceOutputImage, img_size);
  cudaMemcpy(deviceInputImage, hostInputImage, img_size, cudaMemcpyHostToDevice);
  dim3 dimGrid(ceil(imageWidth / TILE_WIDTH), ceil(imageHeight / TILE_WIDTH), 1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  conv2d <<< dimGrid, dimBlock >>>(deviceOutputImage, deviceInputImage, imageWidth, imageHeight, arg);
  cudaMemcpy(hostOutputImage, deviceOutputImage, img_size, cudaMemcpyDeviceToHost);
  write_image_template<int>(argv[2], hostOutputImage, imageWidth, imageHeight);
  cudaFree(deviceInputImage);
  cudaFree(deviceOutputImage);
  free(hostOutputImage);

  return 0;
}
