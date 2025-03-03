//
// Created by Renatus Madrigal on 3/2/2025.
//

#ifndef MANDELBROTSET_INCLUDE_MANDELBROT_MANDELBROTSET_H
#define MANDELBROTSET_INCLUDE_MANDELBROT_MANDELBROTSET_H

#include <complex>
#include <opencv2/core/mat.hpp>
#include "mandelbrot/BaseMandelbrotSet.h"

namespace Mandelbrot {

    class MandelbrotSet : public BaseMandelbrotSet<MandelbrotSet> {
        using Base = BaseMandelbrotSet<MandelbrotSet>;

    public:
        friend class Base;
        constexpr static size_t MAX_ITERATIONS = 1000;
        constexpr static double ESCAPE_RADIUS = 2.0;
        constexpr static size_t COLOR_CYCLE = 50;

        MandelbrotSet() = default;
        MandelbrotSet(const size_t width, const size_t height) : BaseMandelbrotSet(width, height) {}

#ifdef MANDELBROT_SET_TEST
        // TODO: Add test helper.
        friend class MandelbrotSetTest;
#endif

    private:
        [[nodiscard]] cv::Mat generateImpl() const;
        [[nodiscard]] static size_t computeEscapeTime(const std::complex<double> &c);
        [[nodiscard]] static cv::Vec3b computeColor(size_t escape_time);
    };

} // namespace Mandelbrot

#endif // MANDELBROTSET_INCLUDE_MANDELBROT_MANDELBROTSET_H
