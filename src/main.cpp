//
// Created by Renatus Madrigal on 3/2/2025.
//

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <exec/async_scope.hpp>
#include <exec/inline_scheduler.hpp>
#include <exec/static_thread_pool.hpp>
#include <exec/task.hpp>
#include <iostream>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <queue>
#include <thread>
#include "MandelbrotSet.h"
#include "MandelbrotSetCuda.h"
#include "MandelbrotSetMPFR.h"


using namespace cv;
using namespace std;
namespace ex = stdexec;

struct CommandLineArguments {
    size_t width;
    size_t height;
    double x_min, x_max, y_min, y_max;
    bool video;
    double x_center, y_center, xsize, ysize;
    int max_step;
    double zoom_factor, scale_rate;
    bool with_key_frames;
    bool set_output;
    string output;
};

#if ENABLE_CUDA
using DefaultMandelbrotSet = Mandelbrot::MandelbrotSetCuda;
constexpr std::string_view CURRENT_IMPLEMENTATION = "CUDA";
#elif ENABLE_MPFR
using DefaultMandelbrotSet = Mandelbrot::MandelbrotSetMPFR;
constexpr std::string_view CURRENT_IMPLEMENTATION = "MPFR";
#else
using DefaultMandelbrotSet = Mandelbrot::MandelbrotSet;
constexpr std::string_view CURRENT_IMPLEMENTATION = "Pure CPU";
#endif

CommandLineArguments parseArguments(int argc, char **argv_raw) {
    CommandLineArguments args = {
            .width = 2048,
            .height = 2048,
            .x_min = -2.0,
            .x_max = 2.0,
            .y_min = -2.0,
            .y_max = 2.0,
            .video = false,
            .x_center = 0.0,
            .y_center = 0.0,
            .xsize = 4.0,
            .ysize = 5.0,
            .max_step = 10,
            .zoom_factor = 4.0,
            .scale_rate = 1.03,
            .with_key_frames = false,
            .set_output = false,
            .output = "",
    };
    vector<string> argv(argv_raw, argv_raw + argc);
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
            args.x_min = std::stod(argv[i + 1]);
            ++i;
        } else if (argv[i] == "--xmax") {
            assert(i + 1 < argc);
            args.x_max = std::stod(argv[i + 1]);
            ++i;
        } else if (argv[i] == "--ymin") {
            assert(i + 1 < argc);
            args.y_min = std::stod(argv[i + 1]);
            ++i;
        } else if (argv[i] == "--ymax") {
            assert(i + 1 < argc);
            args.y_max = std::stod(argv[i + 1]);
            ++i;
        } else if (argv[i] == "--range") {
            assert(i + 4 < argc);
            args.x_min = std::stod(argv[i + 1]);
            args.x_max = std::stod(argv[i + 2]);
            args.y_min = std::stod(argv[i + 3]);
            args.y_max = std::stod(argv[i + 4]);
            i += 4;
        } else if (argv[i] == "--center") {
            assert(i + 4 < argc);
            args.x_center = std::stod(argv[i + 1]);
            args.y_center = std::stod(argv[i + 2]);
            args.xsize = std::stod(argv[i + 3]);
            args.ysize = std::stod(argv[i + 4]);
            i += 4;
        } else if (argv[i] == "--video") {
            assert(i + 3 < argc);
            args.video = true;
            args.max_step = std::stoi(argv[i + 1]);
            args.zoom_factor = std::stod(argv[i + 2]);
            args.scale_rate = std::stod(argv[i + 3]);
            i += 3;
        } else if (argv[i] == "--with-keyframes") {
            args.with_key_frames = true;
        } else if (argv[i] == "-o" || argv[i] == "--output") {
            assert(i + 1 < argc);
            args.set_output = true;
            args.output = argv[i + 1];
            ++i;
        } else if (argv[i] == "--help") {
            cout << "A simple Mandelbrot set generator" << endl;
            cout << "Current implementation: " << CURRENT_IMPLEMENTATION << endl;
            cout << endl;
            cout << "Usage: " << argv[0] << " [options]" << endl;
            cout << "Options:" << endl;
            cout << "  -o / --output <filename>                       Set the filename of the output file" << endl;
            cout << "  --resolution <width> <height>                  Set the resolution of the image" << endl;
            cout << "  --width <width>                                Set the width of the image" << endl;
            cout << "  --height <height>                              Set the height of the image" << endl;
            cout << "  --xmin <xmin>                                  Set the minimum x value" << endl;
            cout << "  --xmax <xmax>                                  Set the maximum x value" << endl;
            cout << "  --ymin <ymin>                                  Set the minimum y value" << endl;
            cout << "  --ymax <ymax>                                  Set the maximum y value" << endl;
            cout << "  --range <xmin> <xmax> <ymin> <ymax>            Set the range of x and y values" << endl;
            cout << "  --center <xcenter> <ycenter> <xsize> <ysize>   Set the center and size for video" << endl;
            cout << "  --video <max_step> <zoom_factor> <scale_rate>  Generate a zooming animation" << endl;
            cout << "  --with-keyframes                               Generate keyframes for the video" << endl;
            cout << "  --help                                         Display this help message" << endl;
            exit(0);
        }
    }

    return args;
}

vector<Mat> precomputeScaleMatrices(const Point2f &center, double scale_rate, int frames) {
    vector<Mat> scale_matrices(frames);
    double factor = 1.0;
    for (int i = 0; i < frames; ++i) {
        scale_matrices[i] = getRotationMatrix2D(center, 0, factor);
        factor *= scale_rate;
    }
    return scale_matrices;
}

void generateVideo(const CommandLineArguments &args) {
    // Require arguments: max_step, zoom_factor, scale_rate, x_center, y_center, xsize, width, height
    // Some constants for the zooming animation. We may make them configurable later.
    const int max_step = args.max_step;
    const double zoom_factor = args.zoom_factor;
    const double scale_rate = args.scale_rate;


    // Seahorse Valley. We need a more precise center for this.
    // double xcenter = -0.74525, ycenter = 0.12265;
    const double xcenter = args.x_center, ycenter = args.y_center;

    // Image resolution
    const int width = args.width, height = args.height;

    auto filename = args.set_output ? args.output : "MandelbrotSet.mp4";

    cout << "Generating video..." << endl;
    cout << "Resolution: " << width << " x " << height << endl;
    cout << "Center: " << xcenter << ", " << ycenter << endl;
    cout << "Max step: " << max_step << endl;
    cout << fixed << setprecision(2);
    cout << "Zoom factor: " << zoom_factor << endl;
    cout << "Scale rate: " << scale_rate << endl;

    const int frame_count = ceil(log(zoom_factor) / log(scale_rate));

    DefaultMandelbrotSet mandelbrot_set;
    mandelbrot_set.setResolution(width, height);

    VideoWriter writer;
    writer.open(filename, VideoWriter::fourcc('h', 'v', 'c', 'l'), 30, Size(width, height), true);
    Point2f center(width / 2.0, height / 2.0);


    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    const auto scale_matrices = precomputeScaleMatrices(center, scale_rate, frame_count);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    cout << "Time taken to precompute scale matrices: " << diff.count() << " seconds" << endl;

    vector<Mat> frames(frame_count);

    for (uint64_t i = 0, factor = (1 << i); i < max_step; ++i, factor *= zoom_factor) {

        mandelbrot_set.setCenter(xcenter, ycenter, args.xsize / factor);

        start = std::chrono::steady_clock::now();
        const auto image = mandelbrot_set.generate();
        end = std::chrono::steady_clock::now();
        diff = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        cout << "Time taken to generate frame " << i + 1 << ": " << diff.count() << " seconds" << endl;


        if (args.with_key_frames) {
            filename = std::format("frames/MandelbrotSetKeyFrame{}.png", i + 1);
            imwrite(filename, image);
        }

        if (i == max_step - 1)
            break;

        cout << "Start generating frames between " << i + 1 << " and " << i + 2 << endl;

        start = std::chrono::steady_clock::now();

#if ENABLE_OPENMP
#pragma omp parallel for
#endif
        for (int idx = 0; idx < frame_count; ++idx) {
            warpAffine(image, frames[idx], scale_matrices[idx], Size(width, height));
        }

        for (auto &frame: frames) {
            writer.write(frame);
        }

        end = std::chrono::steady_clock::now();
        diff = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        cout << "Time taken to generate frames between " << i + 1 << " and " << i + 2 << ": " << diff.count()
             << " seconds" << endl;
    }

    writer.release();
}

struct RAIITracker {
    RAIITracker() {
        id = getId();
        //        std::println(cout, "RAIITracker {} created on thread {}", id, this_thread::get_id());
    };
    ~RAIITracker() {
        //        std::println(cout, "RAIITracker {} destroyed on thread {}", id, this_thread::get_id());
    }
    RAIITracker(const RAIITracker &other) {
        id = getId();
        std::println(cout, "RAIITracker {} copied to object {} on thread {} ", other.id, id, this_thread::get_id());
        if (other.id < 0) {
            std::println(cerr, "ERROR: Copying from a moved object");
            std::terminate();
        }
    }
    RAIITracker(RAIITracker &&other) {
        id = getId();
        //        std::println(cout, "RAIITracker {} moved on thread {} to object {}", other.id, this_thread::get_id(),
        //        id);
        other.id = -other.id;
    }
    static int getId(bool reset = false) {
        static int counter_ = 0;
        if (reset) {
            counter_ = 0;
            return 0;
        }
        return counter_++;
    }

    int id;
};

template<typename T>
struct AsyncChannel {
    template<typename... Args>
    void send(Args... args) {
        queue.emplace(std::forward<T>(args)...);
    }
    ex::sender auto receive() {
        return ex::just() | ex::then([this]() -> std::optional<T> {
                   if (queue.empty()) {
                       return std::nullopt;
                   } else {
                       T val = queue.front();
                       queue.pop();
                       return val;
                   }
               });
    }

private:
    std::queue<T> queue;
};

#define CURRENT_TIME                                                                                                   \
    (std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - start).count())

// TODO: Implement async video generation as an experiment
void asyncGenerateVideo(const CommandLineArguments &args) {
    // Require arguments: max_step, zoom_factor, scale_rate, x_center, y_center, xsize, width, height
    // Some constants for the zooming animation. We may make them configurable later.
    const int max_step = args.max_step;
    const double zoom_factor = args.zoom_factor;
    const double scale_rate = args.scale_rate;


    // Seahorse Valley. We need a more precise center for this.
    // double xcenter = -0.74525, ycenter = 0.12265;
    const double xcenter = args.x_center, ycenter = args.y_center;

    // Image resolution
    const int width = args.width, height = args.height;
    const double xsize = args.xsize, ysize = args.ysize;

    auto filename = args.set_output ? args.output : "MandelbrotSet.mp4";

    cout << "Generating video..." << endl;
    cout << "Resolution: " << width << " x " << height << endl;
    cout << "Center: " << xcenter << ", " << ycenter << endl;
    cout << "Initial size: " << xsize << " x " << ysize << endl;
    cout << "Max step: " << max_step << endl;
    cout << fixed << setprecision(2);
    cout << "Zoom factor: " << zoom_factor << endl;
    cout << "Scale rate: " << scale_rate << endl;

    constexpr size_t worker_count = 4;
    size_t current_step = 0;
    exec::static_thread_pool pool(4);
    exec::static_thread_pool io_pool(4);
    auto sched = pool.get_scheduler();

    exec::async_scope scope;

    using namespace std::chrono_literals;
    RAIITracker::getId(true);

    cout << "Main thread: " << this_thread::get_id() << endl;

    auto start = std::chrono::steady_clock::now();

    // TODO: Currently we use sleep_for to simulate the generation flow.

    auto keyframeGenerator = [&](int step) {
        std::println(cout, "Generating keyframe {} on thread {} at {}s", step, this_thread::get_id(), CURRENT_TIME);
        // Pretend to generate keyframes
        std::this_thread::sleep_for(2s);
        std::println(cout, "Keyframes generated on thread {} at {}s", this_thread::get_id(), CURRENT_TIME);
        // Call constructor of RAIITracker to track the memory usage
        return RAIITracker{};
    };

    auto imgWriter = [&](auto &&) {
        std::println(cout, "Writing image on thread {} at {}s", this_thread::get_id(), CURRENT_TIME);
        std::this_thread::sleep_for(1.5s);
        std::println(cout, "Image written on thread {} at {}s", this_thread::get_id(), CURRENT_TIME);
    };

    AsyncChannel<RAIITracker> channel;

    auto videoGenerator = [&]() -> exec::task<void> { // NOLINT(*-static-accessed-through-instance)
        while (true) {
            auto value = co_await channel.receive();
            if (value) {
                std::println(cout, "Received value {} on thread {} at {}s", value->id, this_thread::get_id(),
                             CURRENT_TIME);
                // Pretend Video generation
                this_thread::sleep_for(1s);
                std::println(cout, "Video generated on thread {} at {}s", this_thread::get_id(), CURRENT_TIME);
                co_await ex::just();
            } else {
                co_await ex::just();
            }
        }
    };

    scope.spawn(ex::starts_on(sched, videoGenerator()));

    for (auto step: views::iota(0, max_step)) {
        auto res = keyframeGenerator(step);
        channel.send(res);
        scope.spawn(ex::starts_on(io_pool.get_scheduler(), ex::just(std::move(res)) | ex::then(imgWriter)));
    }

    scope.request_stop();

    ex::sync_wait(scope.on_empty());

    std::println(cout, "All work done on thread {}", this_thread::get_id());
}

void generateImage(const CommandLineArguments &args) {
    DefaultMandelbrotSet mandelbrot_set;
    mandelbrot_set.setResolution(args.width, args.height);
    mandelbrot_set.setXRange(args.x_min, args.x_max).setYRange(args.y_min, args.y_max);

    cout << fixed << setprecision(2);
    cout << "Resolution: " << mandelbrot_set.getWidth() << " x " << mandelbrot_set.getHeight() << endl;
    cout << "XRange: " << mandelbrot_set.getXMin() << " - " << mandelbrot_set.getXMax() << endl;
    cout << "YRange: " << mandelbrot_set.getYMin() << " - " << mandelbrot_set.getYMax() << endl;

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    const auto image_cuda = mandelbrot_set.generate();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    cout << "Time taken to generate the image using CUDA: " << diff.count() << " seconds" << endl;

    auto filename = args.set_output ? args.output : "MandelbrotSetCuda.png";
    imwrite(filename, image_cuda);
}

int main(int argc, char **argv) {
    setLogLevel(utils::logging::LogLevel::LOG_LEVEL_SILENT);

    CommandLineArguments args = parseArguments(argc, argv);

#if 0
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    cout << "Current implementation: " << CURRENT_IMPLEMENTATION << endl;

    if (args.video) {
        generateVideo(args);
    } else {
        generateImage(args);
    }

    std::chrono::steady_clock::time_point finish = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::duration<double>>(finish - begin);

    cout << "Total time: " << total_time.count() << " seconds" << endl;
#endif

    asyncGenerateVideo(args);

    return 0;
}
