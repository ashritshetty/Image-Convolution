#include <mpi.h>

#include "convolution_2d_mpi_cuda.h"
#include "utilities.h"

//void compute_gpu(int *hostPartInImage, int *hostPartOutImage, int imageWidth, int imageHeight, int arg);

int main (int argc, char **argv)
{
  int rank, size, imageWidth, imageHeight, kernelType, pixel;
  int ELE_PER_PROC, ELE_PER_SCRA, KERNEL_DIM;
  int *hostInputImage, *hostOutputImage, *hostPartInImage, *hostPartOutImage;

  int kernel_sharp[9] = {-1,-1,-1,-1, 9,-1,-1,-1,-1};
  int kernel_smooth[25] = {0,1,2,1,0,1,4,8,4,1,2,8,16,8,2,1,4,8,4,1,0,1,2,1,0};

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if(rank == 0)
  {
    if (argc != 4)
    {
      printf("Usage    : ./convolution_2d_mpi <input> <output> <kernel>\n");
      printf("<kernel> : 0 - SHARP\n         : 1 - SMOOTH\n");
      exit(1);
    }
  }

  kernelType = atoi(argv[3]);

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

  ELE_PER_PROC = (imageHeight*imageWidth)/size;
  hostPartInImage = (int *)malloc(ELE_PER_PROC * sizeof(int));
  hostPartOutImage = (int *)malloc(ELE_PER_PROC * sizeof(int));

  MPI_Scatter(hostInputImage, ELE_PER_PROC, MPI_INT, hostPartInImage, ELE_PER_PROC, MPI_INT, 0, MPI_COMM_WORLD);

  int i,j;
/*  if(rank == 0)
  {
    for(i = 0; i < imageHeight/size; i++)
    {
      for(j = 0; j < imageWidth; j++)
      {
        printf("%d ", hostInputImage[i*imageWidth + j]);
      }
      printf("\n");
    }	
  } */

  compute_gpu(hostPartInImage, hostPartOutImage, imageWidth, (imageHeight/size), kernelType);

  MPI_Barrier(MPI_COMM_WORLD);

/*  if(rank == 0)
  {
    for(i = 0; i < imageHeight/size; i++)
    {
      for(j = 0; j < imageWidth; j++)
      {
        printf("%d ", hostPartOutImage[i*imageWidth + j]);
      }
      printf("\n");
    }	
  }


*/
  MPI_Gather(hostPartOutImage, ELE_PER_PROC, MPI_INT, hostOutputImage, ELE_PER_PROC, MPI_INT, 0, MPI_COMM_WORLD);

/*    if(rank == 0)
  {
    for(i = 0; i < imageHeight; i++)
    {
      for(j = 0; j < imageWidth; j++)
      {
        printf("%d ", hostOutputImage[i*imageWidth + j]);
      }
      printf("\n");
    }	
  }
*/


  free(hostPartInImage);
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
