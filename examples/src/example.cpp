#include "example.hpp"

using std::cout;
using std::endl;

int main()
{
	std::vector<float> h_dataset_1(10000, 1.0f);
	std::vector<float> h_dataset_2(10000, 1.5f);
	std::vector<float> h_matches(10000, 0.0f);

	int stride = 1;
	int k = 2, n_iter = 100;
	
	CUDADatum * d_dataset_1 = new CUDADatum(
		h_dataset_1,
		stride
	);

	CUDADatum * d_dataset_2 = new CUDADatum(
		h_dataset_2,
		stride
	);

	CUDADatum * d_matches = new CUDADatum(
		h_matches,
		stride
	);

	/*
		1060 specs from device query
		Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
		Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
	*/
	// initialize dim of grid and thread blocks 
	h_cudaKMeans(d_dataset_1, d_dataset_2, d_matches);

	float * res = d_matches->download();
	cout<< res[0];
	
	while (true) {

	}
	return 0;
}
