#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {
#include "convolution_2d_mpi_cuda.h"
}

#define TILE_WIDTH 16
#define SH_MEM_WIDTH_3 18
#define SH_MEM_WIDTH_5 20

__global__ void conv2d3(int *outputImage, int *inputImage, int width, int height, int arg)
{
  int i, j, pixel;
  __shared__ int shared_image[SH_MEM_WIDTH_3][SH_MEM_WIDTH_3];
  int kernel[3][3] = {-1,-1,-1,-1, 9,-1,-1,-1,-1};
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  shared_image[(x%TILE_WIDTH)+1][(y%TILE_WIDTH)+1] = inputImage[y*width+x];
  __syncthreads();

  if(x%TILE_WIDTH == 0)
    shared_image[0][(y%TILE_WIDTH+1)] = shared_image[1][(y%TILE_WIDTH)+1];

  if(x%TILE_WIDTH == TILE_WIDTH-1)
    shared_image[SH_MEM_WIDTH_3-1][(y%TILE_WIDTH)+1] = shared_image[SH_MEM_WIDTH_3-2][(y%TILE_WIDTH)+1];

  if(y%TILE_WIDTH == 0)
    shared_image[(x%TILE_WIDTH)+1][0] = shared_image[(x%TILE_WIDTH)+1][1];

  if(y%TILE_WIDTH == TILE_WIDTH-1)
    shared_image[(x%TILE_WIDTH)+1][SH_MEM_WIDTH_3-1] = shared_image[(x%TILE_WIDTH)+1][SH_MEM_WIDTH_3-2];
  __syncthreads();

  if((x%TILE_WIDTH)+(y%TILE_WIDTH) == 0){
    shared_image[0][0] = shared_image[1][1];
    shared_image[SH_MEM_WIDTH_3-1][SH_MEM_WIDTH_3-1] = shared_image[SH_MEM_WIDTH_3-2][SH_MEM_WIDTH_3-2];
    shared_image[0][SH_MEM_WIDTH_3-1] = shared_image[1][SH_MEM_WIDTH_3-2];
    shared_image[SH_MEM_WIDTH_3-1][0] = shared_image[SH_MEM_WIDTH_3-2][1];
  }
  __syncthreads();

  pixel = 0;
  for(i = 0; i < 3; i++){
    for(j = 0; j < 3; j++){
      pixel = pixel + shared_image[(x%TILE_WIDTH)+i][(y%TILE_WIDTH)+j]*kernel[i][j];
    }
  }
  if(x < width && y < height){
    outputImage[y*width+x] = abs(pixel);
  }
}

__global__ void conv2d5(int *outputImage, int *inputImage, int width, int height, int arg)
{
  int i, j, pixel;
  __shared__ int shared_image[SH_MEM_WIDTH_5][SH_MEM_WIDTH_5];
  int kernel[5][5] = {0,1,2,1,0,1,4,8,4,1,2,8,16,8,2,1,4,8,4,1,0,1,2,1,0};
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  shared_image[(x%TILE_WIDTH)+2][(y%TILE_WIDTH)+2] = inputImage[y*width+x];
  __syncthreads();


  if(x%TILE_WIDTH == 0){
    shared_image[0][(y%TILE_WIDTH+2)] = shared_image[2][(y%TILE_WIDTH)+2];
    shared_image[1][(y%TILE_WIDTH+2)] = shared_image[2][(y%TILE_WIDTH)+2];
  }
  
  if(x%TILE_WIDTH == TILE_WIDTH-1){
    shared_image[SH_MEM_WIDTH_5-1][(y%TILE_WIDTH)+2] = shared_image[SH_MEM_WIDTH_5-3][(y%TILE_WIDTH)+2];
    shared_image[SH_MEM_WIDTH_5-2][(y%TILE_WIDTH)+2] = shared_image[SH_MEM_WIDTH_5-3][(y%TILE_WIDTH)+2];
  }

  if(y%TILE_WIDTH == 0){
    shared_image[(x%TILE_WIDTH)+2][0] = shared_image[(x%TILE_WIDTH)+2][2];
    shared_image[(x%TILE_WIDTH)+2][1] = shared_image[(x%TILE_WIDTH)+2][2];
  }

  if(y%TILE_WIDTH == TILE_WIDTH-1){
    shared_image[(x%TILE_WIDTH)+2][SH_MEM_WIDTH_5-1] = shared_image[(x%TILE_WIDTH)+2][SH_MEM_WIDTH_5-3];
    shared_image[(x%TILE_WIDTH)+2][SH_MEM_WIDTH_5-2] = shared_image[(x%TILE_WIDTH)+2][SH_MEM_WIDTH_5-3];
  }
  __syncthreads();

  if((x%TILE_WIDTH)+(y%TILE_WIDTH) == 0){
    shared_image[0][0] = shared_image[2][2];
    shared_image[1][1] = shared_image[2][2];
    shared_image[SH_MEM_WIDTH_5-1][SH_MEM_WIDTH_5-1] = shared_image[SH_MEM_WIDTH_5-3][SH_MEM_WIDTH_5-3];
    shared_image[SH_MEM_WIDTH_5-2][SH_MEM_WIDTH_5-2] = shared_image[SH_MEM_WIDTH_5-3][SH_MEM_WIDTH_5-3];
    shared_image[0][SH_MEM_WIDTH_5-1] = shared_image[2][SH_MEM_WIDTH_5-3];
    shared_image[1][SH_MEM_WIDTH_5-2] = shared_image[2][SH_MEM_WIDTH_5-3];
    shared_image[SH_MEM_WIDTH_5-1][0] = shared_image[SH_MEM_WIDTH_5-3][2];
    shared_image[SH_MEM_WIDTH_5-2][1] = shared_image[SH_MEM_WIDTH_5-3][2];
  }
  __syncthreads();

  pixel = 0;
  for(i = 0; i < 5; i++){
    for(j = 0; j < 5; j++){
      pixel = pixel + shared_image[(x%TILE_WIDTH)+i][(y%TILE_WIDTH)+j]*kernel[i][j];
    }
  }
  if(x < width && y < height){
    outputImage[y*width+x] = abs(pixel)/80;
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
  if(arg == 0){
    conv2d3 <<< dimGrid, dimBlock >>>(deviceOutputImage, deviceInputImage, imageWidth, imageHeight, arg);
  }
  else{
    conv2d5 <<< dimGrid, dimBlock >>>(deviceOutputImage, deviceInputImage, imageWidth, imageHeight, arg);
  }
  cudaMemcpy(hostOutputImage, deviceOutputImage, img_size, cudaMemcpyDeviceToHost);
  cudaFree(deviceInputImage);
  cudaFree(deviceOutputImage);
}
