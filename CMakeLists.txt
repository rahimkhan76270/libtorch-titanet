cmake_minimum_required(VERSION 3.10)
project(MyNeuralNet)

set(CMAKE_PREFIX_PATH "/home/rahim-khan/libtorch")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/include ${TORCH_INCLUDE_DIRS})

add_executable(main main.cpp)
add_library(subblock src/subblock.cpp)
add_library(prologue src/prologue.cpp)
add_library(SqueezeExcitation src/SqueezeExcitation.cpp)

target_link_libraries(main "${TORCH_LIBRARIES}" subblock prologue SqueezeExcitation)
