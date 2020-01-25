cmake_minimum_required (VERSION 3.8)

project ("cuda_kmeans" VERSION 1.1 DESCRIPTION "CUDA Implementation of KMeans Algorithm" LANGUAGES CXX CUDA)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()
# Include sub-projects.
# main library
add_subdirectory ("src")
# executables
add_subdirectory ("examples")