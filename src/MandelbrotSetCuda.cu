//
// Created by Renatus Madrigal on 3/3/2025.
//

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "mandelbrot/MandelbrotSetCuda.h"

#define CHECK_CUDA(err)                                                                                                \
    do {                                                                                                               \
        if (err != cudaSuccess) {                                                                                      \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));                                              \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

namespace Mandelbrot {
    constexpr auto MAX_ITERATIONS = MandelbrotSetCuda::MAX_ITERATIONS;

    __constant__ static uchar3 colors[MAX_ITERATIONS + 1];
    static bool initialized = false;

    void initialize() {
        if (initialized)
            return;
        // TODO: Figure out why this is necessary
        static uchar3 colors_temp[MAX_ITERATIONS + 1];
        initialized = true;
        cv::Mat hsv(1, MAX_ITERATIONS, CV_8UC3);
        for (int n = 0; n < MAX_ITERATIONS; ++n) {
            const double hue = 180 * fmod(n * 0.3, 1.0);
            hsv.at<cv::Vec3b>(0, n) = cv::Vec3b(hue, 255, 255);
        }
        cvtColor(hsv, hsv, cv::COLOR_HSV2BGR);

        for (int n = 0; n < MAX_ITERATIONS; ++n) {
            auto color = hsv.at<cv::Vec3b>(0, n);
            colors_temp[n] = {color[0], color[1], color[2]};
        }
        colors_temp[MAX_ITERATIONS] = {0, 0, 0};

        CHECK_CUDA(cudaMemcpyToSymbol(colors, colors_temp, sizeof(colors_temp)));
    }

    __global__ void mandelbrotKernel(uchar3 *image, // NOLINT
                                     size_t width, size_t height, // NOLINT
                                     double x_min, double x_max, // NOLINT
                                     double y_min, double y_max) {
        constexpr auto ESCAPE_RADIUS_SQ = MandelbrotSetCuda::ESCAPE_RADIUS_SQ;
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) {
            return;
        }
        const double cr = x_min + (x_max - x_min) * x / width;
        const double ci = y_min + (y_max - y_min) * y / height;

        double zr = 0.0, zi = 0.0;
        int n = 0;
        while (n < MAX_ITERATIONS) {
            const double zr2 = zr * zr;
            const double zi2 = zi * zi;
            if (zr2 + zi2 > ESCAPE_RADIUS_SQ)
                break;

            const double zr_temp = zr2 - zi2 + cr;
            zi = 2 * zr * zi + ci;
            zr = zr_temp;
            ++n;
        }
        const int idx = y * width + x;
        image[idx] = colors[n];
    }

    cv::Mat MandelbrotSetCuda::generateImpl() const {
        initialize();

        uchar3 *d_image;
        CHECK_CUDA(cudaMalloc(&d_image, width_ * height_ * sizeof(uchar3)));

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);

        mandelbrotKernel<<<grid, block>>>(d_image, width_, height_, x_min_, x_max_, y_min_, y_max_);
        CHECK_CUDA(cudaGetLastError());

        cv::Mat image(height_, width_, CV_8UC3);
        CHECK_CUDA(cudaMemcpy(image.data, d_image, width_ * height_ * sizeof(uchar3), cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(d_image));

        return image;
    }


} // namespace Mandelbrot
