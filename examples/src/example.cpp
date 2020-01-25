#include "example.hpp"

using std::cout;
using std::endl;

int main()
{
	std::vector<float> h_dataset = {
		float{1}, float{1}, float{1}
	};

	int k = 2, n_iter = 100;

	CUDADatum * dataset = new CUDADatum(h_dataset.size(), h_dataset.data());
	/*
		1060 specs from device query
		Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
		Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
	*/
	// initialize dim of grid and thread blocks 

	h_cudaKMeans(*dataset);

	while (true) {

	}
	return 0;
}
