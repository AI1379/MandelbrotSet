//
// Created by Renatus Madrigal on 3/21/2025.
//

#ifndef MANDELBROTSET_SRC_MANDELBROTSET_ALGORITHM_H
#define MANDELBROTSET_SRC_MANDELBROTSET_ALGORITHM_H

/**
 * @file Algorithm.h
 */

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace Mandelbrot {

    /**
     * @brief Detects the high gradient in the matrix.
     * @param matrix The matrix to detect the high gradient.
     * @return The matrix with the high gradient detected.
     */
    cv::Mat detectHighGradient(const cv::Mat &matrix);

} // namespace Mandelbrot

#endif // MANDELBROTSET_SRC_MANDELBROTSET_ALGORITHM_H
