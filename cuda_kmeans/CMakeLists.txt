# CMakeList.txt : CMake project for cuda_kmeans, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Add source to this project's executable.
add_executable (cuda_kmeans "cuda_kmeans.cpp" "cuda_kmeans.h")

# TODO: Add tests and install targets if needed.
