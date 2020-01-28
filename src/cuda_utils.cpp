#include "cuda_kmeans/cuda_utils.hpp"

void error(const char *s) {
	perror(s);
	assert(0);
	exit(-1);
}


void check_error(cudaError_t status) {
	cudaError_t prev_status = cudaGetLastError();

	if (status != cudaSuccess) {
		const char *s = cudaGetErrorString(status);
		char buffer[256];
		printf("CUDA Error: %s\n", s);
		assert(0);
		snprintf(buffer, 256, "CUDA Error: %s", s);
		error(buffer);
	}

	if (prev_status != cudaSuccess) {
		const char *s = cudaGetErrorString(status);
		char buffer[256];
		printf("CUDA Error Prev: %s\n", s);
		assert(0);
		snprintf(buffer, 256, "CUDA Error Prev: %s", s);
		error(buffer);
	}

	cudaDeviceSynchronize();

}
	