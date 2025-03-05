# Mandelbrot Set

This is a simple implementation of the Mandelbrot set in C++ with OpenCV. Just for fun.

It was inspired by the math soft course during the Spring 2025 semester at ZJU.

## Build Pre-requisites

Currently, this project is only tested on Windows with `MSVC` compiler. The CPU version is also tested on WSL Arch with
gcc.

- CMake
- OpenCV
- Any modern C++ compiler support OpenMP
- CUDA (optional) (recommended)
- MPFR (optional)

## Build Options

| Option          | Description               | Default |
|-----------------|---------------------------|---------|
| `ENABLE_CUDA`   | Build with CUDA support   | ON      |
| `ENABLE_OPENMP` | Build with OpenMP support | ON      |
| `ENABLE_MPFR`   | Build with MPFR support   | OFF     |

## Usage

Assuming that the executable is named `mandelbrot`. You can run `mandelbrot --help` to see the usage.
