# Mandelbrot Set

This is a simple implementation of the Mandelbrot set in C++ with OpenCV. Just for fun.

It was inspired by the math soft course during the Spring 2025 semester at ZJU.

If you like this project, please give me a star. Thanks! :smile:

## Features

- [x] Add more color schemes
- [x] Algorithm to auto detect infinite area
- [x] Asynchronous I/O for image saving and video generation
- [x] CPM for package management
- [x] CI/CD on GitHub using GitHub actions.
- [x] Better commandline interface
- [ ] ~~BMP output without third-party library~~
- [ ] ~~Benchmark~~
- [ ] Report and documentation
- [ ] Import AI to create memes based on the Mandelbrot set
- [ ] Julia set

### Bonus Points

- [x] Submitted before the sixth week
- [x] With a clear code structure with detailed comments and documentation
- [x] Using CMake for building
- [x] Better visualization using OpenCV
- [x] More than one color scheme
- [x] Generating video zooming in
- [x] Using [P2300 -  `std::execution`](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html) for
  parallelization
- [x] Detecting infinite area using gradient descent algorithm

## Binary Release

For convenience, you can find binary releases in GitHub action artifacts. These binary releases are built with CUDA and
OpenCV support. You may need to install OpenCV and CUDA runtime to run the binary.

## Build

### Pre-requisites

Currently, this project is only tested on Windows with `MSVC` compiler. The CPU version is also tested on WSL Arch with
gcc.

- CMake
- OpenCV (recommended)
- Any modern C++ compiler support OpenMP and C++20. C++23 will be better.
- CUDA (optional) (recommended)
- MPFR (optional) (not recommended)

We use [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake) for package management. It will automatically download and
build other dependencies,
including [NVIDIA/stdexec](https://github.com/NVIDIA/stdexec), [ericniebler/range-v3](https://github.com/ericniebler/range-v3)
and [fmtlib/fmt](https://github.com/fmtlib/fmt). All these dependencies are header-only libraries, and they will be or
have already been part of C++ standard library in C++23 or later.

### Options

| Option              | Description                          | Default |
|---------------------|--------------------------------------|---------|
| `ENABLE_CUDA`       | Build with CUDA support              | ON      |
| `ENABLE_OPENMP`     | Build with OpenMP support            | ON      |
| `ENABLE_MPFR`       | Build with MPFR support              | OFF     |
| `ENABLE_STDEXEC`    | Build with stdexec support           | ON      |
| `ENABLE_OPENCV`     | Build with OpenCV support            | ON      |
| `ENABLE_EXT_DOUBLE` | Use `ExtendedDouble` for calculation | ON      |

See [Note](#note) for more information.

### Build with CMake

Just like any CMake project. For example:

```shell 
mkdir build
cd build
cmake .. 
cmake --build . -j$(nproc)
```

It is recommended using `ninja` as the generator for better performance.

### Note

- It is recommended to build with OpenCV for better visualization and performance. Besides, video output requires
  OpenCV.
- CUDA is recommended for GPU acceleration.
- `ExtendedDouble` is a custom implementation of arbitrary precision floating point number. More details can be found
  in [report.tex](doc/report.tex).
- `stdexec` is a C++ library for asynchronous execution as part of C++23. However, it is not yet supported by most
  compilers. Thus, the implementation by NVIDIA is used.
- OpenMP is used for parallelization. It is recommended to enable it for better performance. Most of the modern C++
  compilers support OpenMP, so it should be fine.
- MPFR is used for arbitrary precision floating point number. It is optional and not recommended for normal usage. The
  performance is severely degraded.
- A paper is written for this project in the `doc` directory.
- Because this repository is published on GitHub as well, the paper is written in English, with a Chinese version
  [report-cn.tex](doc/report-cn.tex). For the same reason, the signature and school ID are removed from the paper.

## Usage

Assuming that the executable is named `mandelbrot`. You can run `mandelbrot --help` to see the usage.

```text
A simple Mandelbrot set generator

Usage: mandelbrot [options]
Options:
    -o / --output <filename>                       Set the filename of the output file
    --resolution <width> <height>                  Set the resolution of the image
    --width <width>                                Set the width of the image
    --height <height>                              Set the height of the image
    --xmin <xmin>                                  Set the minimum x value
    --xmax <xmax>                                  Set the maximum x value
    --ymin <ymin>                                  Set the minimum y value
    --ymax <ymax>                                  Set the maximum y value
    --range <xmin> <xmax> <ymin> <ymax>            Set the range of x and y values
    --center <xcenter> <ycenter> <xsize> <ysize>   Set the center and size for video
    --video <max_step> <zoom_factor> <scale_rate>  Generate a zooming animation
    --with-keyframes                               Generate keyframes for the video
    --auto-detect                                  Automatically detect keyframes
    --show-grid                                    Show grid on keyframes
    --help                                         Display this help message

Default values:
    width: 2048
    height: 2048
    xmin: -2.0
    xmax: 2.0
    ymin: -2.0
    ymax: 2.0

Example:
    mandelbrot --resolution 2048 2048 --xmin -2.0 --xmax 2.0 --ymin -2.0 --ymax 2.0
    mandelbrot --video 100 4.0 1.03 --center -0.74525 0.12265 4.0 5.0
```
