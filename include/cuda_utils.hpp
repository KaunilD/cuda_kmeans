#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H
#endif

#include "cuda.h"
#include <cuda_runtime.h>

#ifndef LIBS_H
#include "libs.hpp"
#endif


void error(const char *);
void check_error(cudaError_t );