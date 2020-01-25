#include "cuda_datum.hpp"

__global__ void d_cudaKMeans(float * datum){
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	datum[id]+=datum[id];
}

void h_cudaKMeans(CUDADatum & datum){
	// 32 = warp size. shared memory is 
	// permitted between threads in a block
	int blockSize = 32*10; // number of threads in a block
	int gridSize = datum.size/blockSize; // number of blocks;
	d_cudaKMeans<<<dim3(gridSize), dim3(blockSize)>>>(datum.d_buffer);
}