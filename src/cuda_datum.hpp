#include "cuda.h"
#include "cuda_utils.hpp"

class CUDADatum {
public:

	float3 * d_buffer;
	int size;

	CUDADatum(int _size, float3 *data): size(sizeof(float3)*_size) {
		cudaError_t status = cudaMalloc(&d_buffer, size);
		CUDAUtils::check_error(status);
		status = cudaMemcpy(&d_buffer, data, size, cudaMemcpyHostToDevice);
		CUDAUtils::check_error(status);
	}

	void clear() {
		cudaMemset(d_buffer, 0, size);
		size = 0;
	}

	~CUDADatum() {
		clear();
		cudaFree(d_buffer);
	};


};