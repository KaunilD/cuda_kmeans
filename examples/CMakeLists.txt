cmake_minimum_required (VERSION 3.8)

message(INFO ": Building example")

include_directories(${PROJECT_SOURCE_DIR}/include/)
add_executable(example "example.hpp" "example.cpp")

target_link_libraries(example libcudakmeans)
