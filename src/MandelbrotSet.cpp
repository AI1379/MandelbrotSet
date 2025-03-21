//
// Created by Renatus Madrigal on 3/2/2025.
//

#include "MandelbrotSet.h"
#include <opencv2/imgproc.hpp>
#include "BaseMandelbrotSet.h"

namespace Mandelbrot {
    size_t MandelbrotSet::computeEscapeTime(const std::complex<double> &c) {
        std::complex<double> z(0.0, 0.0);
        for (auto i = 0u; i < MAX_ITERATIONS; ++i) {
            z = z * z + c;
            if (std::norm(z) > ESCAPE_RADIUS * ESCAPE_RADIUS) {
                return i;
            }
        }
        return MAX_ITERATIONS;
    }

    cv::Mat MandelbrotSet::generateRawMatrixImpl() const {
        cv::Mat image(height_, width_, CV_32FC1);

        const double xscale = (x_max_ - x_min_) / width_;
        const double yscale = (y_max_ - y_min_) / height_;

#if ENABLE_OPENMP
#pragma omp parallel for
#endif
        for (auto y = 0; y < height_; ++y) {
            for (auto x = 0; x < width_; ++x) {
                double xcoord = x_min_ + x * xscale;
                double ycoord = y_min_ + y * yscale;
                std::complex<double> c(xcoord, ycoord);
                size_t escape_time = computeEscapeTime(c);
                image.at<float>(y, x) = static_cast<float>(escape_time);
            }
        }

        return image;
    }


} // namespace Mandelbrot
