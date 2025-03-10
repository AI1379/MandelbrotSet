//
// Created by Renatus Madrigal on 3/6/2025.
//

#include "ImageWriter.h"
#include <opencv2/imgcodecs.hpp>

namespace Mandelbrot {

    void SyncImageWriter::write(cv::Mat &&image, const std::string &filename) { cv::imwrite(filename, image); }

} // namespace Mandelbrot
