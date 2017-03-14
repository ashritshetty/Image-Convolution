#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include "utilities.h"

int main (int argc, char **argv)
{
  int rank, size, imageWidth, imageHeight, kernelType, pixel, i, j, k, l, x, y;
  int ELE_PER_PROC, ELE_PER_SCRA, KERNEL_DIM;
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
    kernel = &kernel_sharp[0];
  }
  else
  {
    KERNEL_DIM = 5;
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

  ELE_PER_PROC = (imageHeight/size)*imageWidth;
  ELE_PER_SCRA = ((imageHeight/size)+(2*(KERNEL_DIM/2)))*(imageWidth+(2*(KERNEL_DIM/2)));
  hostPartInImage = (int *)malloc(ELE_PER_PROC * sizeof(int));
  hostScratchImage = (int *)malloc(ELE_PER_SCRA * sizeof(int));
  hostPartOutImage = (int *)malloc(ELE_PER_PROC * sizeof(int));

  MPI_Scatter(hostInputImage, ELE_PER_PROC, MPI_INT, hostPartInImage, ELE_PER_PROC, MPI_INT, 0, MPI_COMM_WORLD);

  for(i = 0; i < sizeof(hostScratchImage)/sizeof(int); i++){
    hostScratchImage[i] = 0;
  }
  for(i = 0; i < imageHeight/size; i++){
    for(j = 0; j < imageWidth; j++){
      hostScratchImage[(i+(KERNEL_DIM/2)*(imageWidth+KERNEL_DIM-1))+((KERNEL_DIM/2)+j)] = hostInputImage[(i*(imageWidth-1))+j];
    }
  }
  for(i = 0; i < 2; i++){
    for(j = (KERNEL_DIM/2); j < imageWidth; j++){
      hostScratchImage[(i*(imageWidth+KERNEL_DIM-1))+j] = hostScratchImage[((i+2)*(imageWidth+KERNEL_DIM-1))+j];
    }
  }
  for(i = (imageHeight/size); i < (imageHeight/size)+2; i++){
    for(j = (KERNEL_DIM/2); j < imageWidth; j++){
      hostScratchImage[(i*(imageWidth+KERNEL_DIM-1))+j] = hostScratchImage[((i-2)*(imageWidth+KERNEL_DIM-1))+j];
    }
  }

  for(i = 0; i < imageWidth-1+(KERNEL_DIM/2); i++){
    for(j = 0; j < imageHeight-1+(KERNEL_DIM/2); j++){
      pixel = 0;
      x = i;
      for(k = 0; k < KERNEL_DIM; k++){
        y = j;
        for(l = 0; l < KERNEL_DIM; l++){
          pixel = pixel + (kernel[k*KERNEL_DIM+l] * hostScratchImage[x*(imageWidth+KERNEL_DIM)+y]);
          y++;
        }
        x++;
      }
      hostPartOutImage[i*imageWidth+j] = pixel;
    }
  }

  MPI_Gather(hostPartOutImage, ELE_PER_PROC, MPI_INT, hostOutputImage, ELE_PER_PROC, MPI_INT, 0, MPI_COMM_WORLD);

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
