cmake_minimum_required(VERSION 3.16)

project(sum_sin)

set(CMAKE_CXX_STANDARD 20)

option(USE_FLOAT "Use float instead of double" OFF)

if(USE_FLOAT)
    add_compile_definitions(DATA_TYPE=float)
else()
    add_compile_definitions(DATA_TYPE=double)
endif()

add_executable(sum_sin source/main.cpp)