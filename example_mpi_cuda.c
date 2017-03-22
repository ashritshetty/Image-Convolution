#include <mpi.h>

#include "convolution_2d_mpi_cuda.h"
#include "utilities.h"

int main (int argc, char **argv)
{
  int rank, size, imageWidth, imageHeight, kernelType, i;
  int ELE_PER_PROC, KERNEL_DIM;
  int *hostInputImage, *hostOutputImage, *hostPartInImage, *hostPartOutImage;
  double t1, t2, t3, t4;

  int kernel_sharp[9] = {-1,-1,-1,-1, 9,-1,-1,-1,-1};
  int kernel_smooth[25] = {0,1,2,1,0,1,4,8,4,1,2,8,16,8,2,1,4,8,4,1,0,1,2,1,0};

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if(rank == 0){
    if (argc != 4){
      printf("Usage    : ./example_mpi_cuda <input> <output> <kernel>\n");
      printf("<kernel> : 0 - SHARP\n         : 1 - SMOOTH\n");
      exit(1);
    }
  }

  kernelType = atoi(argv[3]);
  
  if(rank == 0)
    t1 = MPI_Wtime();

  if(rank == 0){
    read_image_template(argv[1], &hostInputImage, &imageWidth, &imageHeight);
    for(i = 1; i < size; i++){
      MPI_Send(&imageWidth, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      MPI_Send(&imageHeight, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
    hostOutputImage = (int *)malloc(imageWidth * imageHeight * sizeof(int));
  }
  else{
    MPI_Recv(&imageWidth, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&imageHeight, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  ELE_PER_PROC = (imageHeight*imageWidth)/size;
  hostPartInImage = (int *)malloc(ELE_PER_PROC * sizeof(int));
  hostPartOutImage = (int *)malloc(ELE_PER_PROC * sizeof(int));

  MPI_Scatter(hostInputImage, ELE_PER_PROC, MPI_INT, hostPartInImage, ELE_PER_PROC, MPI_INT, 0, MPI_COMM_WORLD);

  if(rank == 0)
    t2 = MPI_Wtime();
    
  compute_gpu(hostPartInImage, hostPartOutImage, imageWidth, (imageHeight/size), kernelType);

  if(rank == 0)
    t3 = MPI_Wtime();

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Gather(hostPartOutImage, ELE_PER_PROC, MPI_INT, hostOutputImage, ELE_PER_PROC, MPI_INT, 0, MPI_COMM_WORLD);
   
  free(hostPartInImage);
  free(hostPartOutImage);
  
  if(rank == 0){
    write_image_template(argv[2], hostOutputImage, imageWidth, imageHeight);
    free(hostInputImage);
    free(hostOutputImage);
  }

  if(rank == 0){
    t4 = MPI_Wtime();
    //printf("%f\n", t4-t1);
    printf("GPU Time %f\n", t3-t2);
    printf("MPI Time %f\n", (t4-t1)-(t3-t2));
    printf("Total Time %f\n", t4-t1);
  }

  MPI_Finalize();
  return 0;
}
