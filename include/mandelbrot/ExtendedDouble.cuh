//
// Created by Renatus Madrigal on 3/4/2025.
//

#ifndef MANDELBROTSET_INCLUDE_MANDELBROT_EXPANDDOUBLE_CUH
#define MANDELBROTSET_INCLUDE_MANDELBROT_EXPANDDOUBLE_CUH

namespace Mandelbrot {
    struct __align__(16) ExtendedDouble {
        double mantissa;
        int exponent;
    };

    __device__ ExtendedDouble operator+(ExtendedDouble a, ExtendedDouble b);

} // namespace Mandelbrot

#endif // MANDELBROTSET_INCLUDE_MANDELBROT_EXPANDDOUBLE_CUH
