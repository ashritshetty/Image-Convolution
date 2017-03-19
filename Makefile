all: convolution_2d_mpi_cuda_host.o convolution_2d_mpi_cuda_device.o
	mpicc -o convolution_2d_mpi_cuda convolution_2d_mpi_cuda_host.o -L /software/cuda-toolkit/8.0.44/lib64 -lcudart convolution_2d_mpi_cuda_device.o

convolution_2d_mpi_cuda.o: convolution_2d_mpi_cuda_host.c
	mpicc -c convolution_2d_mpi_cuda_host.c

convolution_2d_mpi_cuda.o: convolution_2d_mpi_cuda_device.cu
	nvcc --ptxas-options=-v -c convolution_2d_mpi_cuda_device.cu

clean:
	rm -rf *.o convolution_2d_mpi_cuda
