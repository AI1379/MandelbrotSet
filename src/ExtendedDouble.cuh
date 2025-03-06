//
// Created by Renatus Madrigal on 3/4/2025.
//

#ifndef MANDELBROTSET_INCLUDE_MANDELBROT_EXPANDDOUBLE_CUH
#define MANDELBROTSET_INCLUDE_MANDELBROT_EXPANDDOUBLE_CUH

#include <algorithm>
#include <cmath>
#include <iostream>

namespace Mandelbrot {
    using std::abs;

    __host__ __device__ __forceinline__ double max(double a, double b) { return a > b ? a : b; }
    __host__ __device__ __forceinline__ int max(int a, int b) { return a > b ? a : b; }

    struct __align__(16) ExtendedDouble {
        double mantissa;
        int exponent;

        __host__ __device__ ExtendedDouble() : mantissa(0.0), exponent(0) {}

        __host__ __device__ explicit ExtendedDouble(double val);

        __host__ __device__ void normalize();

        __host__ __device__ explicit operator double() const { return ldexp(mantissa, exponent); }

        __host__ __device__ ExtendedDouble operator*(const ExtendedDouble &rhs) const;

        __host__ __device__ __forceinline__ ExtendedDouble operator*(int rhs) const {
            if (rhs == 2) {
                ExtendedDouble result = *this;
                result.exponent += 1;
                return result;
            }
            return *this * ExtendedDouble(rhs);
        }

        __host__ __device__ ExtendedDouble operator/(const ExtendedDouble &rhs) const;

        __host__ __device__ ExtendedDouble operator+(const ExtendedDouble &rhs) const;

        __host__ __device__ ExtendedDouble operator-(const ExtendedDouble &rhs) const;


        __host__ __device__ __forceinline__ bool operator==(const ExtendedDouble &rhs) const {
            return exponent == rhs.exponent && mantissa == rhs.mantissa;
        }

        __host__ __device__ __forceinline__ bool operator>(const ExtendedDouble &rhs) const {
            if (exponent != rhs.exponent)
                return exponent > rhs.exponent;
            return mantissa > rhs.mantissa;
        }

        __host__ __device__ __forceinline__ bool operator>(double rhs) const { return *this > ExtendedDouble(rhs); }

        __host__ __device__ __forceinline__ ExtendedDouble operator<<(int shift) const {
            ExtendedDouble result = *this;
            result.exponent += shift;
            return result;
        }

        __host__ __device__ __forceinline__ ExtendedDouble operator>>(int shift) const {
            ExtendedDouble result = *this;
            result.exponent -= shift;
            return result;
        }

        friend std::ostream &operator<<(std::ostream &os, const ExtendedDouble &ed);
    };

} // namespace Mandelbrot

#endif // MANDELBROTSET_INCLUDE_MANDELBROT_EXPANDDOUBLE_CUH
