#include "cuda_kmeans/cuda_datum.hpp"

__device__ float d_l2norm(float f1, float f2){
	return f1-f2;
}

__global__ void d_cudaKMeans(float * dataset_1, float * dataset_2, float * matches, int length){
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id >= length){
		return;
	}
	float last_distance_matched = INT_MAX, curr_distance_matched = 0;
	
	matches[id] = 0.0;
	for(size_t i = 0; i < length; ++i){
		if (i == id){
			continue;
		}
		curr_distance_matched = d_l2norm(dataset_1[id], dataset_2[i]);
		if ( last_distance_matched > curr_distance_matched ){
			matches[id] = i;
			last_distance_matched = curr_distance_matched;
		}
	}

	return;
}

void h_cudaKMeans(CUDADatum * datum_1, CUDADatum * datum_2, CUDADatum * matches){
	// 32 = warp size. shared memory is 
	// permitted between threads in a block
	int blockSize = 32*10; // number of threads in a block
	int gridSize = (datum_1->length + blockSize - 1)/blockSize; // number of blocks;

	d_cudaKMeans<<<gridSize, blockSize>>>(
		datum_1->d_in_buffer, datum_2->d_in_buffer, matches->d_out_buffer, datum_1->length
	);

	cudaDeviceSynchronize();
	
}