cmake_minimum_required(VERSION 3.10)

project(Zora LANGUAGES CXX CUDA)

set(SOURCES
    ./core/Vector.h 
    ./core/Matrix.h
    ./main/main.cu
)

add_executable(Zora ${SOURCES})