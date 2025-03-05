cmake_minimum_required(VERSION 3.30)
project(MandelbrotSet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

option(ENABLE_CUDA "Enable CUDA" ON)
message(STATUS "CUDA support: ${ENABLE_CUDA}")
if (ENABLE_CUDA)
    enable_language(CUDA)
endif ()

option(ENABLE_MPFR "Enable MPFR" OFF)
message(STATUS "MPFR support: ${ENABLE_MPFR}")

option(ENABLE_OPENMP "Enable OpenMP" ON)
message(STATUS "OpenMP support: ${ENABLE_OPENMP}")

add_subdirectory(${PROJECT_SOURCE_DIR}/src)