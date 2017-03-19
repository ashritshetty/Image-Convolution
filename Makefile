all: example_mpi_cuda.o example_kernel.o
	mpic++ -o example_mpi_cuda example_kernel.o -L /software/cuda-toolkit/8.0.44/lib64 -lcudart example_mpi_cuda.o

example_mpi_cuda.o: example_mpi_cuda.cpp
	mpic++ -c example_mpi_cuda.cpp

example_kernel.o: example_kernel.cu
	nvcc --ptxas-options=-v -c example_kernel.cu

clean:
	rm -rf *.o example_mpi_cuda
