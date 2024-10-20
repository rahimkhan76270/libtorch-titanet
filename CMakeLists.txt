cmake_minimum_required(VERSION 3.10)
project(MyNeuralNet)

# Set the required C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the path for Libtorch and Pybind11
set(CMAKE_PREFIX_PATH "/home/rahim-khan/libtorch")
# set(pybind11_DIR "/home/rahim-khan/miniconda3/envs/ai_repo/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")

# Use the new FindPython module and ensure all components are found
# set(PYBIND11_FINDPYTHON ON)  # Enable finding Python
# find_package(Python REQUIRED COMPONENTS Development Interpreter)

# Find required packages
find_package(Torch REQUIRED)
# find_package(pybind11 REQUIRED)

# Include directories
# include_directories(${Python_INCLUDE_DIRS})

# Find the sndfile library
find_library(SNDFILE_LIBRARIES NAMES sndfile)
if(NOT SNDFILE_LIBRARIES)
    message(FATAL_ERROR "libsndfile not found")
endif()

# find_library(TORCHAUDIO_LIB torchaudio HINTS /usr/local/torchaudio/lib)

# Add executable and libraries
add_executable(main main.cpp)
add_library(net src/net.cpp)
add_library(read_audio src/read_audio.cpp)
add_library(spectrogram src/spectrogram.cpp)
add_library(dataset src/dataset.cpp)

# Link libraries
target_link_libraries(main 
    PRIVATE 
    "${TORCH_LIBRARIES}" 
    ${TORCHAUDIO_LIB}
    net 
    read_audio 
    ${SNDFILE_LIBRARIES} 
    spectrogram 
    # pybind11::module 
    # ${Python_LIBRARIES}
    dataset
    ${SNDFILE_LIBRARIES} 
)


# Ensure CMake can find the Torch headers
include_directories(${CMAKE_SOURCE_DIR}/include ${TORCH_INCLUDE_DIRS})

# Set C++ standard for the target
set_property(TARGET main PROPERTY CXX_STANDARD 17)