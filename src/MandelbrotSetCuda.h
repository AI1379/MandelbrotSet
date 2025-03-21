//
// Created by Renatus Madrigal on 3/3/2025.
//

#ifndef MANDELBROTSET_INCLUDE_MANDELBROT_MANDELBROTSETCUDA_CUH
#define MANDELBROTSET_INCLUDE_MANDELBROT_MANDELBROTSETCUDA_CUH

#include <opencv2/core.hpp>
#include "BaseMandelbrotSet.h"

namespace Mandelbrot {
    class MandelbrotSetCuda : public BaseMandelbrotSet<MandelbrotSetCuda> {
        using Base = BaseMandelbrotSet<MandelbrotSetCuda>;

    public:
        friend Base;

        constexpr static int MAX_ITERATIONS = 1000;
        constexpr static int BLOCK_SIZE = 16;
        constexpr static double ESCAPE_RADIUS = 2.0;
        constexpr static double ESCAPE_RADIUS_SQ = ESCAPE_RADIUS * ESCAPE_RADIUS;

    private:
        [[nodiscard]] cv::Mat generateRawMatrixImpl() const;
    };
} // namespace Mandelbrot

#endif // MANDELBROTSET_INCLUDE_MANDELBROT_MANDELBROTSETCUDA_CUH
