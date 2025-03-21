//
// Created by Renatus Madrigal on 3/2/2025.
//

#ifndef MANDELBROTSET_INCLUDE_MANDELBROT_MANDELBROTSET_H
#define MANDELBROTSET_INCLUDE_MANDELBROT_MANDELBROTSET_H

#include <complex>
#include <opencv2/core/mat.hpp>
#include "BaseMandelbrotSet.h"

namespace Mandelbrot {

    class MandelbrotSet : public BaseMandelbrotSet<MandelbrotSet> {
        using Base = BaseMandelbrotSet<MandelbrotSet>;

    public:
        friend Base;

        MandelbrotSet() = default;
        MandelbrotSet(const size_t width, const size_t height) : BaseMandelbrotSet(width, height) {}

    private:
        [[nodiscard]] cv::Mat generateRawMatrixImpl() const;
        [[nodiscard]] static size_t computeEscapeTime(const std::complex<double> &c);
    };

} // namespace Mandelbrot

#endif // MANDELBROTSET_INCLUDE_MANDELBROT_MANDELBROTSET_H
