//
// Created by Renatus Madrigal on 3/21/2025.
//

#ifndef MANDELBROTSET_SRC_MANDELBROTSET_ALGORITHM_H
#define MANDELBROTSET_SRC_MANDELBROTSET_ALGORITHM_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace Mandelbrot {

    cv::Mat detectHighGradient(const cv::Mat &matrix);

} // namespace Mandelbrot

#endif // MANDELBROTSET_SRC_MANDELBROTSET_ALGORITHM_H
