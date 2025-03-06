# Build instructions

## Prerequisites

As mentioned in the [README](README.md), you need to have the following installed:

- CMake
- C++ compiler with C++20 support, for example:
    - clang 13+
    - gcc 11+
    - MSVC 19.29+
- Ninja (optional, but recommended)
- CUDA (optional, but recommended)

Besides, we also need the following libraries:

- [OpenCV](https://opencv.org/)
- [stdexec](https://github.com/NVIDIA/stdexec)
- [MPFR](https://www.mpfr.org/)
- [GMP](https://gmplib.org/) (required by MPFR)

Note that the CUDA version of OpenCV is **NOT** necessary. We only use CUDA for computation, not for image processing.

### Installations for these libraries

- OpenCV: Follow the instructions on the [official website](https://opencv.org/). You can also install it via package
  managers like `apt` or `pacman`.
- stdexec: Just clone the repository to any directory you like.
- MPFR: Download the source code from the [official website](https://www.mpfr.org/), and compile it yourself. Package
  managers is also an option. Note that on Windows you may need `msys2` or `mingw` to compile it. **You need to install
  GMP before building MPFR.**

## Environment variables for building with CMake

Here are some environment variables that you need to define to find third-party libraries properly:

| Variable      | Description                                  |
|---------------|----------------------------------------------|
| `OpenCV_DIR`  | Directory that contains `OpenCVConfig.cmake` |
| `STDEXEC_DIR` | Path to the root of `stdexec`                |
| `MPFR_DIR`    | Directory that contains `include/mpfr.h`     |

## Build with CMake

Several options are available when building with CMake:

| Option              | Description                          | Default |
|---------------------|--------------------------------------|---------|
| `ENABLE_CUDA`       | Build with CUDA support              | ON      |
| `ENABLE_OPENMP`     | Build with OpenMP support            | ON      |
| `ENABLE_MPFR`       | Build with MPFR support              | OFF     |
| `ENABLE_STDEXEC`    | Build with stdexec support           | ON      |
| `ENABLE_OPENCV`     | Build with OpenCV support            | ON      |
| `ENABLE_EXT_DOUBLE` | Use `ExtendedDouble` for calculation | ON      |

You can enable or disable these options by passing `-D<option>=ON/OFF` to CMake. For example:

```shell
cmake .. -GNinja -DENABLE_CUDA=OFF -DENABLE_MPFR=ON
```
