# cmake_minimum_required(VERSION 3.10)
# project(MyNeuralNet)

# set(CMAKE_PREFIX_PATH "/home/rahim-khan/libtorch")
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(pybind11_DIR "/home/rahim-khan/miniconda3/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")
# set(Python3_LIBRARIES "/home/rahim-khan/miniconda3/lib/libpython3.12.so")
# set(Python3_INCLUDE_DIRS "/home/rahim-khan/miniconda3/include/python3.12")
# find_package(pybind11 REQUIRED)
# find_package(Python3 COMPONENTS Development REQUIRED)
# include_directories(${Python3_INCLUDE_DIRS})

# find_package(Torch REQUIRED)
# include_directories(${CMAKE_SOURCE_DIR}/include ${TORCH_INCLUDE_DIRS})
# # include_directories("/home/rahim-khan/Work/essentia/src/essentia")
# find_library(SNDFILE_LIBRARIES NAMES sndfile)
# if(NOT SNDFILE_LIBRARIES)
#     message(FATAL_ERROR "libsndfile not found")
# endif()

# add_executable(main main.cpp)
# add_library(net src/net.cpp)
# add_library(read_audio src/read_audio.cpp)
# add_library(spectrogram src/spectrogram.cpp)

# target_link_libraries(main "${TORCH_LIBRARIES}" net read_audio ${SNDFILE_LIBRARIES} spectrogram)
# target_link_libraries(main PRIVATE pybind11::module ${Python3_LIBRARIES})

cmake_minimum_required(VERSION 3.10)
project(MyNeuralNet)

# Set the required C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the path for Libtorch and Pybind11
set(CMAKE_PREFIX_PATH "/home/rahim-khan/libtorch")
set(pybind11_DIR "/home/rahim-khan/miniconda3/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")

# Use the new FindPython module and ensure all components are found
set(PYBIND11_FINDPYTHON ON)  # Enable finding Python
find_package(Python REQUIRED COMPONENTS Development Interpreter)

# Find required packages
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

# Include directories
include_directories(${Python_INCLUDE_DIRS})

# Find the sndfile library
find_library(SNDFILE_LIBRARIES NAMES sndfile)
if(NOT SNDFILE_LIBRARIES)
    message(FATAL_ERROR "libsndfile not found")
endif()

# Add executable and libraries
add_executable(main main.cpp)
add_library(net src/net.cpp)
add_library(read_audio src/read_audio.cpp)
add_library(spectrogram src/spectrogram.cpp)

# Link libraries
target_link_libraries(main 
    PRIVATE 
    "${TORCH_LIBRARIES}" 
    net 
    read_audio 
    ${SNDFILE_LIBRARIES} 
    spectrogram 
    pybind11::module 
    ${Python_LIBRARIES}
)

# Ensure CMake can find the Torch headers
include_directories(${CMAKE_SOURCE_DIR}/include ${TORCH_INCLUDE_DIRS})

# Set C++ standard for the target
set_property(TARGET main PROPERTY CXX_STANDARD 17)
