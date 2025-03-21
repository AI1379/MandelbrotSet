//
// Created by Renatus Madrigal on 3/21/2025.
//

#include "Algorithm.h"

namespace Mandelbrot {

    cv::Mat detectHighGradient(const cv::Mat &matrix) {
        constexpr static double GRADIENT_THRESHOLD = 0.5;

        cv::Mat normalized;
        cv::normalize(matrix, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);

        cv::Mat grad_x, grad_y;
        cv::Sobel(normalized, grad_x, CV_32F, 1, 0);
        cv::Sobel(normalized, grad_y, CV_32F, 0, 1);

        cv::Mat abs_grad_x, abs_grad_y, grad_mag;
        cv::convertScaleAbs(grad_x, abs_grad_x);
        cv::convertScaleAbs(grad_y, abs_grad_y);
        cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad_mag);

        double min_val, max_val;
        cv::minMaxLoc(grad_mag, &min_val, &max_val);
        double threshold = min_val + (max_val - min_val) * GRADIENT_THRESHOLD;

        cv::Mat mask = grad_mag;
        cv::threshold(grad_mag, mask, threshold, 255, cv::THRESH_BINARY);
        return mask;
    }

} // namespace Mandelbrot
