all: example_mpi_cuda.o example_kernel.o
	mpicc -o example_mpi_cuda example_kernel.o -L /software/cuda-toolkit/8.0.44/lib64 -lstdc++ -lcudart example_mpi_cuda.o

example_mpi_cuda.o: example_mpi_cuda.c
	mpicc -c example_mpi_cuda.c

example_kernel.o: example_kernel.cu
	nvcc --ptxas-options=-v -c example_kernel.cu

clean:
	rm -rf *.o example_mpi_cuda
