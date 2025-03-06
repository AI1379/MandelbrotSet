//
// Created by Renatus Madrigal on 3/5/2025.
//

#ifndef MANDELBROTSET_INCLUDE_MANDELBROT_MANDELBROTSETMPFR_H
#define MANDELBROTSET_INCLUDE_MANDELBROT_MANDELBROTSETMPFR_H

#include <opencv2/core/mat.hpp>
#include "BaseMandelbrotSet.h"

namespace Mandelbrot {

    class MandelbrotSetMPFR : public BaseMandelbrotSet<MandelbrotSetMPFR> {
        using Base = BaseMandelbrotSet<MandelbrotSetMPFR>;

    public:
        friend Base;
        constexpr static size_t MAX_ITERATIONS = 1000;
        constexpr static double ESCAPE_RADIUS = 2.0;
        constexpr static size_t COLOR_CYCLE = 50;

        MandelbrotSetMPFR() = default;
        MandelbrotSetMPFR(const size_t width, const size_t height) : BaseMandelbrotSet(width, height) {}

    private:
        [[nodiscard]] cv::Mat generateImpl() const;
    };

} // namespace Mandelbrot

#endif // MANDELBROTSET_INCLUDE_MANDELBROT_MANDELBROTSETMPFR_H
