#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include "utilities.h"

int main (int argc, char **argv)
{
  int rank, size, imageWidth, imageHeight, scratchWidth, scratchHeight, kernelType, pixel, i, j, k, l, x, y, rowIndex, colIndex;
  int ELE_PER_PROC, ELE_PER_SCRA, KERNEL_DIM, KERNEL_RADIUS, DIVIDE;
  int *hostInputImage, *hostOutputImage, *hostPartInImage, *hostScratchImage, *hostPartOutImage, *kernel;

  int kernel_sharp[9] = {-1,-1,-1,-1, 9,-1,-1,-1,-1};
  int kernel_smooth[25] = {0,1,2,1,0,1,4,8,4,1,2,8,16,8,2,1,4,8,4,1,0,1,2,1,0};

  if (argc != 4)
  {
    printf("Usage    : ./convolution_2d_mpi <input> <output> <kernel>\n");
    printf("<kernel> : 0 - SHARP\n         : 1 - SMOOTH\n");
    exit(1);
  }

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  kernelType = atoi(argv[3]);

  if(kernelType == 0)
  {
    KERNEL_DIM = 3;
    KERNEL_RADIUS = 1;
    DIVIDE = 1;
    kernel = &kernel_sharp[0];
  }
  else
  {
    KERNEL_DIM = 5;
    KERNEL_RADIUS = 2;
    DIVIDE = 80;
    kernel = &kernel_smooth[0];
  }

  if(rank == 0)
  {
    read_image_template(argv[1], &hostInputImage, &imageWidth, &imageHeight);
    MPI_Send(&imageWidth, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    MPI_Send(&imageHeight, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    hostOutputImage = (int *)malloc(imageWidth * imageHeight * sizeof(int));
  }
  else
  {
    MPI_Recv(&imageWidth, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&imageHeight, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  scratchWidth = imageWidth + KERNEL_DIM - 1;
  scratchHeight = (imageHeight / size) + KERNEL_DIM - 1;
  ELE_PER_PROC = (imageHeight/size)*imageWidth;
  ELE_PER_SCRA = ((imageHeight/size)+(KERNEL_DIM-1))*(imageWidth+(KERNEL_DIM-1));
  hostPartInImage = (int *)malloc(ELE_PER_PROC * sizeof(int));
  hostScratchImage = (int *)malloc(ELE_PER_SCRA * sizeof(int));
  hostPartOutImage = (int *)malloc(ELE_PER_PROC * sizeof(int));

  MPI_Scatter(hostInputImage, ELE_PER_PROC, MPI_INT, hostPartInImage, ELE_PER_PROC, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  for(i = 0; i < ELE_PER_SCRA; i++){
    hostScratchImage[i] = 0;
  }
  for(i = 0; i < imageHeight/size; i++){
    for(j = 0; j < imageWidth; j++){
      hostScratchImage[((i+KERNEL_RADIUS)*scratchWidth)+(KERNEL_RADIUS+j)] = hostPartInImage[(i*imageWidth)+j];
    }
  }

  for(i = 0; i < KERNEL_RADIUS; i++){
    for(j = KERNEL_RADIUS; j < imageWidth+KERNEL_RADIUS; j++){
      hostScratchImage[(i*scratchWidth)+j] = hostScratchImage[((i+KERNEL_RADIUS)*scratchWidth)+j];
    }
  }
  for(i = (imageHeight/size)+KERNEL_RADIUS; i < (imageHeight/size)+KERNEL_DIM-1; i++){
    for(j = KERNEL_RADIUS; j < imageWidth+KERNEL_RADIUS; j++){
      hostScratchImage[(i*scratchWidth)+j] = hostScratchImage[((i-KERNEL_RADIUS)*scratchWidth)+j];
    }
  }
 
  for(i=0; i < (imageHeight/size); i++)
  {
      for(j=0; j < imageWidth; j++)
      {
          pixel = 0;
          for(k=0; k < KERNEL_DIM; k++)
          {
              x = KERNEL_DIM - 1 - k;

              for(l=0; l < KERNEL_DIM; l++)
              {
                  y = KERNEL_DIM - 1 - l;

                  rowIndex = i + k - (KERNEL_DIM/2);
                  colIndex = j + l - (KERNEL_DIM/2);

                  //printf("%d %d\n", rowIndex, colIndex);
                  //if(rowIndex >= 0 && rowIndex < (imageHeight/size)+KERNEL_DIM-1 && colIndex >= 0 && colIndex < imageWidth+KERNEL_DIM-1)
                      pixel += hostScratchImage[(imageWidth+KERNEL_DIM-1) * rowIndex + colIndex] * kernel[KERNEL_DIM * x + y];
              }
          }
	  //printf("i %d j %d\n", i, j);
          hostPartOutImage[imageWidth * i + j] = abs(pixel)/DIVIDE;
      }
  }

  if(rank == 0)
{
  for(i = 0; i < (imageHeight/size); i++)
  {
    for(j = 0; j < imageWidth; j++)
    {
      printf("%d ", hostPartOutImage[i * imageWidth + j]);
    }
    printf("\n");
  }
 }



  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Gather(hostPartOutImage, ELE_PER_PROC, MPI_INT, hostOutputImage, ELE_PER_PROC, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  free(hostPartInImage);
  free(hostScratchImage);
  free(hostPartOutImage);
  if(rank == 0)
  {
    write_image_template(argv[2], hostOutputImage, imageWidth, imageHeight);
    free(hostInputImage);
    free(hostOutputImage);
  }

  MPI_Finalize();
  return 0;
}
