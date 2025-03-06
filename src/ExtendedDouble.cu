//
// Created by Renatus Madrigal on 3/4/2025.
//

#include "ExtendedDouble.cuh"

namespace Mandelbrot {


    __host__ __device__ ExtendedDouble::ExtendedDouble(double val) {
        int exp;
        mantissa = frexp(val, &exp);

        // Scale mantissa to [1.0, 2.0)
        if (mantissa != 0.0) {
            mantissa *= 2.0;
            exp--;
        }
        exponent = exp;
    }

    __host__ __device__ void ExtendedDouble::normalize() {
        if (mantissa == 0.0) {
            exponent = 0;
            return;
        }

        int exp_shift;
        mantissa = frexp(mantissa, &exp_shift);
        mantissa *= 2.0;
        exponent += (exp_shift - 1);
    }

    __host__ __device__ ExtendedDouble ExtendedDouble::operator*(const ExtendedDouble &rhs) const {
        ExtendedDouble result;
        result.mantissa = mantissa * rhs.mantissa;
        result.exponent = exponent + rhs.exponent;
        result.normalize();
        return result;
    }

    __host__ __device__ ExtendedDouble ExtendedDouble::operator/(const ExtendedDouble &rhs) const {
        ExtendedDouble result;
        result.mantissa = mantissa / rhs.mantissa;
        result.exponent = exponent - rhs.exponent;
        result.normalize();
        return result;
    }

    __host__ __device__ ExtendedDouble ExtendedDouble::operator+(const ExtendedDouble &rhs) const {
        if (mantissa == 0.0)
            return rhs;
        if (rhs.mantissa == 0.0)
            return *this;

        int max_exp = max(exponent, rhs.exponent);
        double a = mantissa;
        double b = rhs.mantissa;

        if (exponent < max_exp) {
            a = ldexp(a, -(max_exp - exponent));
        } else if (rhs.exponent < max_exp) {
            b = ldexp(b, -(max_exp - rhs.exponent));
        }

        ExtendedDouble result;
        result.mantissa = a + b;
        result.exponent = max_exp;
        result.normalize();
        return result;
    }

    __host__ __device__ ExtendedDouble ExtendedDouble::operator-(const ExtendedDouble &rhs) const {
        if (rhs.mantissa == 0.0)
            return *this;

        int max_exp = max(exponent, rhs.exponent);
        double a = mantissa;
        double b = rhs.mantissa;

        if (exponent < max_exp) {
            a = ldexp(a, -(max_exp - exponent));
        } else if (rhs.exponent < max_exp) {
            b = ldexp(b, -(max_exp - rhs.exponent));
        }

        ExtendedDouble result;
        result.mantissa = a - b;
        result.exponent = max_exp;
        result.normalize();
        return result;
    }

    std::ostream &operator<<(std::ostream &os, const ExtendedDouble &ed) {
        os << static_cast<double>(ed);
        return os;
    }
} // namespace Mandelbrot
