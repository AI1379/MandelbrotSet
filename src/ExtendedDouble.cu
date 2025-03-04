//
// Created by Renatus Madrigal on 3/4/2025.
//

#include "mandelbrot/ExtendedDouble.cuh"

namespace Mandelbrot {

    __device__ ExtendedDouble operator+(ExtendedDouble a, ExtendedDouble b) {
        if (a.exponent < b.exponent) {
            const ExtendedDouble tmp = a;
            a = b;
            b = tmp;
        }

        int exp_diff = a.exponent - b.exponent;
        if (exp_diff > 53)
            return a;

        double scaled_b_mantissa = __dmul_rd(b.mantissa, __powf(2.0, -exp_diff));
        double new_mantissa = __dadd_rd(a.mantissa, scaled_b_mantissa);

        int shift;
        new_mantissa = frexp(new_mantissa, &shift);
        return {new_mantissa, a.exponent + shift};
    }

} // namespace Mandelbrot
