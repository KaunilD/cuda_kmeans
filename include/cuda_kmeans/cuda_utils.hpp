#include "cuda.h"
#include <cuda_runtime.h>

#include "libs.hpp"


void error(const char * s);
void check_error(cudaError_t status);