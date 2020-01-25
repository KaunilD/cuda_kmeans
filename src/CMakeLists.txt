cmake_minimum_required (VERSION 3.8)

find_package( CUDA REQUIRED )

if(CUDA_FOUND)
	message(STATUS "CUDA library status:")
	message(STATUS "    NVCC_EXECUTABLE: ${CUDA_NVCC_EXECUTABLE}")
	message(STATUS "    INCLUDE_PATH: ${CUDA_INCLUDE_DIRS}")
	message(STATUS "    TOOLKIT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")

	# compared to class settings, we let NVidia's FindCUDA CMake detect 
	# whether to build x64.  
	# Complied for 1060 architecture. Add more from here:
	# http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
	set(CUDA_NVCC_FLAGS "
		-gencode;arch=compute_61,code=sm_61;"
	)
  
	set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

else(CUDA_FOUND)
	message("CUDA is not installed on this system.")
endif()

set( HEADERS 
	${PROJECT_SOURCE_DIR}/include/cuda_kmeans/libs.hpp
	${PROJECT_SOURCE_DIR}/include/cuda_kmeans/cuda_datum.hpp
	${PROJECT_SOURCE_DIR}/include/cuda_kmeans/cuda_kmeans.hpp
	${PROJECT_SOURCE_DIR}/include/cuda_kmeans/cuda_utils.hpp
)

set ( KERNELS
	${PROJECT_SOURCE_DIR}/kernels/kmeans.cu
)

set ( SRCS
	cuda_datum.cpp
	cuda_kmeans.cpp
	cuda_utils.cpp
)

# Add source to this project's executable.
add_library(libcudakmeans STATIC ${KERNELS} ${SRCS} ${HEADERS})

target_include_directories(
	libcudakmeans 
	PUBLIC
		${CUDA_INCLUDE_DIRS}
		$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:include>
)
target_compile_features(libcudakmeans PUBLIC cxx_std_11)
source_group(TREE "${PROJECT_SOURCE_DIR}/include" PREFIX "Header Files" FILES ${HEADERS})