#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "utilities.h"

#define BLOCKSIZE 256
#define SH_MEM_WIDTH 258

__global__ void conv1d(int *outputVector, int *inputVector, int length, int arg)
{
  int i;
  __shared__ int shared_array[SH_MEM_WIDTH];
  int kernel1D[3][3] = {1,0,1,0,1,0,1,2,3};
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int kernel[3];

  for(i = 0; i < 3; i++)
  {
    kernel[i] = kernel1D[arg][i];
  }

  shared_array[(x%BLOCKSIZE)+1] = inputVector[x];
  __syncthreads();

  if(x%BLOCKSIZE == 0 && x != 0)
    shared_array[0] = inputVector[x-1];
  if(x%BLOCKSIZE == BLOCKSIZE-1 && x != length-1)
    shared_array[SH_MEM_WIDTH-1] = inputVector[x+1];
  if(x == 0)
    shared_array[0] = 0;
  if(x == length-1)
    shared_array[SH_MEM_WIDTH-1] = 0;
  __syncthreads();

  if(x < length)
    outputVector[x] = (shared_array[(x%BLOCKSIZE)]*kernel[0]) + (shared_array[(x%BLOCKSIZE)+1]*kernel[1]) + (shared_array[(x%BLOCKSIZE)+2]*kernel[2]);
}

int main (int argc, char **argv)
{
  int vectorLength, vec_size, arg;
  int *hostInputVector, *hostOutputVector;
  int *deviceInputVector, *deviceOutputVector;

  if (argc != 4)
  {
    printf("Usage    : ./convolution_1d <input> <output> <kernel>\n");
    printf("<kernel> : 0 - 101\n         : 1 - 010\n         : 2 - 123\n");
    exit(1);
  }

  arg = atoi(argv[3]);
  read_matrix(argv[1], &vectorLength, &hostInputVector);
  vec_size = vectorLength * sizeof(int);
  hostOutputVector = (int *)malloc(vec_size);
  cudaMalloc((void **)&deviceInputVector, vec_size);
  cudaMalloc((void **)&deviceOutputVector, vec_size);
  cudaMemcpy(deviceInputVector, hostInputVector, vec_size, cudaMemcpyHostToDevice);
  dim3 dimGrid(ceil(vectorLength / BLOCKSIZE), 1, 1);
  dim3 dimBlock(BLOCKSIZE, 1, 1);
  conv1d <<< dimGrid, dimBlock >>>(deviceOutputVector, deviceInputVector, vectorLength, arg);
  cudaMemcpy(hostOutputVector, deviceOutputVector, vec_size, cudaMemcpyDeviceToHost);
  write_matrix(argv[2], &vectorLength, &hostOutputVector);
  cudaFree(deviceInputVector);
  cudaFree(deviceOutputVector);
  free(hostOutputVector);

  return 0;
}
