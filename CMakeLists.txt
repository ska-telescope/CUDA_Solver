cmake_minimum_required(VERSION 3.10.2)

#! BASE SOLVER PROJECT
project(solver CXX CUDA)

#! FLAGS FOR COMPILER
if ("${CMAKE_C_COMPILER_ID}" MATCHES ".*Clang.*" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
	set(CMAKE_C_FLAGS "-Wall -Wextra")
	set(CMAKE_C_FLAGS_DEBUG "-g -O0")
	set(CMAKE_C_FLAGS_RELEASE "-O3")
	set(CMAKE_CXX_FLAGS "-Wall -Wextra")
	set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

#! LOCATE CUDA
find_package(CUDA REQUIRED)

#! FLAGS FOR CUDA
set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS};
	-std=c++11 -arch=sm_60 -Xptxas -v -O3 -restrict -D_FORCE_INLINES -D_DEBUG -lineinfo
)

#! BUILD AND LINK SOLVER
cuda_add_executable(solver main.cpp solver.cu)
target_link_libraries(solver m cufft) # math and CUDA FFT