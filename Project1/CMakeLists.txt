cmake_minimum_required(VERSION 3.10)
project(mpi_project)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER mpic++)

add_executable(mpi_project main.cpp lodepng.cpp)