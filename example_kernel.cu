#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {
#include "convolution_2d_mpi_cuda.h"
}

#define TILE_WIDTH 16
#define SH_MEM_WIDTH 18

//extern "c" void compute_gpu(int *hostInputImage, int *hostOutputImage, int imageWidth, int imageHeight, int arg);

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

  shared_image[(x%TILE_WIDTH)+1][(y%TILE_WIDTH)+1] = inputImage[y*width+x];
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
    outputImage[y*width+x] = abs(pixel)/9;
  }
}

extern "C" void compute_gpu(int *hostInputImage, int *hostOutputImage, int imageWidth, int imageHeight, int arg)
{
  int img_size;
  int *deviceInputImage, *deviceOutputImage;

  img_size = imageWidth * imageHeight * sizeof(int);
  cudaMalloc((void **)&deviceInputImage, img_size);
  cudaMalloc((void **)&deviceOutputImage, img_size);
  cudaMemcpy(deviceInputImage, hostInputImage, img_size, cudaMemcpyHostToDevice);
  dim3 dimGrid(ceil(imageWidth / TILE_WIDTH), ceil(imageHeight / TILE_WIDTH), 1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  conv2d <<< dimGrid, dimBlock >>>(deviceOutputImage, deviceInputImage, imageWidth, imageHeight, arg);
  cudaMemcpy(hostOutputImage, deviceOutputImage, img_size, cudaMemcpyDeviceToHost);
  cudaFree(deviceInputImage);
  cudaFree(deviceOutputImage);
}
