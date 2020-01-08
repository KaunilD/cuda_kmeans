# CMakeList.txt : CMake project for cuda_kmeans, include source and define
# project specific logic here.
#
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


# Add source to this project's executable.
add_executable (cuda_kmeans "cuda_kmeans.cpp" "cuda_kmeans.h")

# TODO: Add tests and install targets if needed.
target_include_directories( cuda_kmeans PUBLIC ${CUDA_INCLUDE_DIRS})
