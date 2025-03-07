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

option(ENABLE_STDEXEC "Enable STDEXEC" ON)
message(STATUS "stdexec support: ${ENABLE_STDEXEC}")

option(ENABLE_OPENCV "Enable OpenCV" ON)
message(STATUS "OpenCV support: ${ENABLE_OPENCV}")

option(ENABLE_EXT_DOUBLE "Enable Extended Double" OFF)
message(STATUS "Extended Double support: ${ENABLE_EXT_DOUBLE}")

add_subdirectory(${PROJECT_SOURCE_DIR}/src)