//
// Created by Renatus Madrigal on 3/6/2025.
//

#ifndef MANDELBROTSET_SRC_IMAGEWRITER_H
#define MANDELBROTSET_SRC_IMAGEWRITER_H

#include <opencv2/core.hpp>
#include <string>
#include <exec/static_thread_pool.hpp>

namespace Mandelbrot {
    class SyncImageWriter {
    public:
        void write(cv::Mat &&image, const std::string &filename);
    };

    class AsyncImageWriter {
    public:
        void write(cv::Mat &&image, const std::string &filename);

    private:
        exec::static_thread_pool thread_pool_;
    };

#if ENABLE_STDEXEC
    using ImageWriter = AsyncImageWriter;
#else
    using ImageWriter = SyncImageWriter;
#endif
} // namespace Mandelbrot

#endif // MANDELBROTSET_SRC_IMAGEWRITER_H
