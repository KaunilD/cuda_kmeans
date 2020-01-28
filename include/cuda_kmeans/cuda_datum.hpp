#include "cuda_utils.hpp"


class CUDADatum {
public:

	float * d_in_buffer{ nullptr };
	float * d_out_buffer{ nullptr };

	float * h_out_buffer{ nullptr };
	
	size_t length{ 0 };
	int bytes{ 0 };
	int stride{ 0 };
	
	CUDADatum(std::vector<float> &data, unsigned int stride);
	~CUDADatum();

	void clear();
	float * download();	

};
