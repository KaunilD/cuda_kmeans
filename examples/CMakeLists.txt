cmake_minimum_required (VERSION 3.8)



find_package( CUDA REQUIRED )
find_package( CUDAKMEANS REQUIRED )

add_executable(
	example "example.hpp" "example.cpp"
)

target_include_directories(example PUBLIC ${CUDAKMEANS_INCLUDE_DIRS})

target_link_libraries(example libcudakmeans)
