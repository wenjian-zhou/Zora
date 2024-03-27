cmake_minimum_required(VERSION 3.10)

project(Zora LANGUAGES CXX CUDA)

set(SOURCES
    ./src/core/Global.h
    ./src/core/Vector.h 
    ./src/core/Matrix.h
)

add_executable(Zora ${SOURCES})