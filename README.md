# Mandelbrot Set

This is a simple implementation of the Mandelbrot set in C++ with OpenCV. Just for fun.

It was inspired by the math soft course during the Spring 2025 semester at ZJU.

If you like this project, please give me a star. Thanks! :smile:

## TODO

- [ ] Add more color schemes
- [ ] Algorithm to auto detect infinite area
- [ ] Asynchronous I/O for image saving and video generation
- [ ] CPM for package management
- [ ] Better commandline interface
- [ ] BMP output without third-party library
- [ ] Benchmark

## Build

### Pre-requisites

Currently, this project is only tested on Windows with `MSVC` compiler. The CPU version is also tested on WSL Arch with
gcc.

- CMake
- OpenCV (recommended)
- [NVIDIA/stdexec](https://github.com/NVIDIA/stdexec) (optional) (recommended)
- Any modern C++ compiler support OpenMP
- CUDA (optional) (recommended)
- MPFR (optional)

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
  in [report.tex](docs/report.tex).
- `stdexec` is a C++ library for asynchronous execution as part of C++23. However, it is not yet supported by most
  compilers. Thus, the implementation by NVIDIA is used.
- OpenMP is used for parallelization. It is recommended to enable it for better performance. Most of the modern C++
  compilers support OpenMP, so it should be fine.
- MPFR is used for arbitrary precision floating point number. It is optional and not recommended for normal usage. The
  performance is severely degraded.
- A paper is written for this project in the `doc` directory.
- Because this repository is published on GitHub as well, the paper is written in English, with a Chinese version
  [report-cn.tex](docs/report-cn.tex). For the same reason, the signature and school ID are removed from the paper.

## Usage

Assuming that the executable is named `mandelbrot`. You can run `mandelbrot --help` to see the usage.
