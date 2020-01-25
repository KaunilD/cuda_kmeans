cmake_minimum_required (VERSION 3.8)

project ("cuda_kmeans"  LANGUAGES CXX CUDA)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")

# Include sub-projects.
add_subdirectory ("src")
add_subdirectory ("examples")