//
// Created by Renatus Madrigal on 3/2/2025.
//

#include <cassert>
#include <chrono>
#include <cstdlib>
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
    double xmin, xmax, ymin, ymax;
    double xcenter, ycenter, xsize;
    bool use_center;
};

CommandLineArguments parseArguments(int argc, char **argv) {
    CommandLineArguments args = {.width = 1024,
                                 .height = 1024,
                                 .xmin = -2.0,
                                 .xmax = 2.0,
                                 .ymin = -2.0,
                                 .ymax = 2.0,
                                 .xcenter = 0.0,
                                 .ycenter = 0.0,
                                 .xsize = 4.0,
                                 .use_center = false};
    for (size_t i = 1; i < argc; i++) {
        if (argv[i] == "--resolution") {
            assert(i + 2 < argc);
            args.width = std::stoi(argv[i + 1]);
            args.height = std::stoi(argv[i + 2]);
            i += 2;
        } else if (argv[i] == "--width") {
            assert(i + 1 < argc);
            args.width = std::stoi(argv[i + 1]);
            ++i;
        } else if (argv[i] == "--height") {
            assert(i + 1 < argc);
            args.height = std::stoi(argv[i + 1]);
            ++i;
        } else if (argv[i] == "--xmin") {
            assert(i + 1 < argc);
            args.xmin = std::stod(argv[i + 1]);
            ++i;
        } else if (argv[i] == "--xmax") {
            assert(i + 1 < argc);
            args.xmax = std::stod(argv[i + 1]);
            ++i;
        } else if (argv[i] == "--ymin") {
            assert(i + 1 < argc);
            args.ymin = std::stod(argv[i + 1]);
            ++i;
        } else if (argv[i] == "--ymax") {
            assert(i + 1 < argc);
            args.ymax = std::stod(argv[i + 1]);
            ++i;
        } else if (argv[i] == "--range") {
            assert(i + 4 < argc);
            args.xmin = std::stod(argv[i + 1]);
            args.xmax = std::stod(argv[i + 2]);
            args.ymin = std::stod(argv[i + 3]);
            args.ymax = std::stod(argv[i + 4]);
            i += 4;
        } else if (argv[i] == "--center") {
            assert(i + 3 < argc);
            args.xcenter = std::stod(argv[i + 1]);
            args.ycenter = std::stod(argv[i + 2]);
            args.xsize = std::stod(argv[i + 3]);
            args.use_center = true;
            i += 3;
        } else if (argv[i] == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --resolution <width> <height>  Set the resolution of the image" << std::endl;
            std::cout << "  --width <width>                Set the width of the image" << std::endl;
            std::cout << "  --height <height>              Set the height of the image" << std::endl;
            std::cout << "  --xmin <xmin>                  Set the minimum x value" << std::endl;
            std::cout << "  --xmax <xmax>                  Set the maximum x value" << std::endl;
            std::cout << "  --ymin <ymin>                  Set the minimum y value" << std::endl;
            std::cout << "  --ymax <ymax>                  Set the maximum y value" << std::endl;
            std::cout << "  --range <xmin> <xmax> <ymin> <ymax> Set the range of x and y values" << std::endl;
            std::cout << "  --center <xcenter> <ycenter> <xsize> Set the center and size of the image" << std::endl;
            std::cout << "  --help                        Display this help message" << std::endl;
            exit(0);
        }
    }

    return args;
}

int main(int argc, char **argv) {
    auto args = parseArguments(argc, argv);

    auto width = args.width;
    auto height = args.height;

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
    mandelbrot_set_cuda.setResolution(width, height);
    if (args.use_center) {
        mandelbrot_set_cuda.setCenter(args.xcenter, args.ycenter, args.xsize);
    } else {
        mandelbrot_set_cuda.setXRange(args.xmin, args.xmax).setYRange(args.ymin, args.ymax);
    }

    start = std::chrono::steady_clock::now();
    const auto image_cuda = mandelbrot_set_cuda.generate();
    end = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Time taken to generate the image using CUDA: " << diff.count() << " seconds" << std::endl;

    imwrite("MandelbrotSetCuda.png", image_cuda);
#endif

    return 0;
}
