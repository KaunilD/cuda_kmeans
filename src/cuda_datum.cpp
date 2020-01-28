#include "cuda_kmeans/cuda_datum.hpp"
#include <iostream>
CUDADatum::CUDADatum(std::vector<float> &data, unsigned int stride) : stride(stride) {
	
	length = data.size();
	bytes = sizeof(float)*length;

	cudaError_t status = cudaMalloc(&d_in_buffer, bytes);
	std::cout << "cudaMalloc " << status << "." << std::endl;
	check_error(status);

	status = cudaMalloc(&d_out_buffer, bytes);
	std::cout << "cudaMalloc " << status << "." << std::endl;
	check_error(status);

	status = cudaMemcpy(d_in_buffer, data.data(), bytes, cudaMemcpyHostToDevice);
	std::cout << "cudaMemcpyHostToDevice " << status << "." << std::endl;
	check_error(status);
}

float * CUDADatum::download() {

	h_out_buffer = { new float[length] {} };

	cudaError_t status = cudaMemcpy(h_out_buffer, d_out_buffer, bytes, cudaMemcpyDeviceToHost);
	std::cout << "cudaMemcpyDeviceToHost " << status << "." <<std::endl;
	check_error(status);
	
	return h_out_buffer;
}

void CUDADatum::clear() {
	cudaMemset(d_out_buffer, 0, bytes);
	bytes = 0;
	stride = 0;
	length = 0;
}

CUDADatum::~CUDADatum() {
	clear();
	cudaFree(d_out_buffer);
};