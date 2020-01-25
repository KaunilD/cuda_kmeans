#include "cuda_utils.hpp"
class CUDADatum {
public:

	float * d_buffer;
	int size;

	CUDADatum(size_t _size, float *data);

	void clear();
	~CUDADatum();
};
