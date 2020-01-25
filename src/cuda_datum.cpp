#include "cuda_datum.hpp"

CUDADatum::CUDADatum(size_t _size, float *data) : size(sizeof(float3)*_size) {
		cudaError_t status = cudaMalloc(&d_buffer, size);
		check_error(status);
		status = cudaMemcpy(&d_buffer, data, size, cudaMemcpyHostToDevice);
		check_error(status);
	}

void CUDADatum::clear() {
		cudaMemset(d_buffer, 0, size);
		size = 0;
	}

CUDADatum::~CUDADatum() {
		clear();
		cudaFree(d_buffer);
};