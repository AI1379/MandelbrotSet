//
// Created by Renatus Madrigal on 3/2/2025.
//

#include <chrono>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "mandelbrot/MandelbrotSet.h"
#include "mandelbrot/MandelbrotSetCuda.h"

using namespace cv;
using namespace std;

struct ZoomParams {
    double xmin, xmax, ymin, ymax;
};

struct CommandLineArguments {
    size_t width;
    size_t height;
};

CommandLineArguments parseCommandLineArguments(int argc, char **argv) {
    CommandLineArguments args{800, 800};
    if (argc > 1) {
        args.width = std::stoi(argv[1]);
    }
    if (argc > 2) {
        args.height = std::stoi(argv[2]);
    }
    return args;
}

int main(int argc, char **argv) {
    auto args = parseCommandLineArguments(argc, argv);

    auto width = args.width;
    auto height = args.height;

    ZoomParams zp{-2.0, 1.0, -1.5, 1.5};
    std::chrono::steady_clock::time_point start, end;
    std::chrono::duration<double> diff{};

#if 0
    Mandelbrot::MandelbrotSet mandelbrot(width, height);

    mandelbrot.setXRange(zp.xmin, zp.xmax).setYRange(zp.ymin, zp.ymax);

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    const auto image = mandelbrot.generate();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Time taken to generate the image: " << diff.count() << " seconds" << std::endl;

    imwrite("MandelbrotSet.png", image);
#endif


#if ENABLE_CUDA
    Mandelbrot::MandelbrotSetCuda mandelbrot_set_cuda;
    mandelbrot_set_cuda.setResolution(width, height).setXRange(zp.xmin, zp.xmax).setYRange(zp.ymin, zp.ymax);

    start = std::chrono::steady_clock::now();
    const auto image_cuda = mandelbrot_set_cuda.generate();
    end = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Time taken to generate the image using CUDA: " << diff.count() << " seconds" << std::endl;

    imwrite("MandelbrotSetCuda.png", image_cuda);
#endif

    return 0;
}
