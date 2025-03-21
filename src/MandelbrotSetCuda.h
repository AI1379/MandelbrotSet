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

        constexpr static int BLOCK_SIZE = 16;

    private:
        [[nodiscard]] cv::Mat generateRawMatrixImpl() const;
    };
} // namespace Mandelbrot

#endif // MANDELBROTSET_INCLUDE_MANDELBROT_MANDELBROTSETCUDA_CUH
