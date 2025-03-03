# Mandelbrot Set

This is a simple implementation of the Mandelbrot set in C++ with OpenCV. Just for fun.

It was inspired by the math soft course during the Spring 2025 semester at ZJU.

## Build Pre-requisites

Currently, this project is only tested on Windows with `MSVC` compiler. However, I believe it should work on Linux as
well.

### Windows

- CMake
- OpenCV
- MSVC (Clang and GCC should work as well, but I haven't tested it)
- CUDA (optional)

## Build Options

| Option          | Description               | Default |
|-----------------|---------------------------|---------|
| `ENABLE_CUDA`   | Build with CUDA support   | ON      |
| `ENABLE_OPENMP` | Build with OpenMP support | ON      |

## Usage

Assuming that the executable is named `mandelbrot`, you can run it with the following command:

```shell
mandelbrot <width> <height>
```

Where `<width>` and `<height>` are the dimensions of the output image. Currently, the range of the Mandelbrot set is not
configurable.