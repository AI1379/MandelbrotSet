//
// Created by Renatus Madrigal on 3/2/2025.
//

#include <opencv2/highgui.hpp>
#include "mandelbrot/MandelbrotSet.h"

using namespace cv;
using namespace std;

// 交互参数
struct ZoomParams {
    double xmin, xmax, ymin, ymax;
    Point start;
    bool dragging;
};

// 鼠标回调函数实现缩放
void on_mouse(int event, int x, int y, int flags, void *param) {
    auto *zp = static_cast<ZoomParams *>(param);

    if (event == EVENT_LBUTTONDOWN) {
        zp->start = Point(x, y);
        zp->dragging = true;
    } else if (event == EVENT_LBUTTONUP && zp->dragging) {
        constexpr double ZOOM_FACTOR = 0.8;
        // 计算新视图区域
        const double dx = (zp->xmax - zp->xmin) * ZOOM_FACTOR;
        const double dy = (zp->ymax - zp->ymin) * ZOOM_FACTOR;

        // 计算新坐标系
        const double new_xmin = zp->xmin + (zp->start.x - dx / 2) * (zp->xmax - zp->xmin) / zp->xmax;
        const double new_xmax = new_xmin + dx;
        const double new_ymin = zp->ymin + (zp->start.y - dy / 2) * (zp->ymax - zp->ymin) / zp->ymax;
        const double new_ymax = new_ymin + dy;

        // 更新参数
        zp->xmin = new_xmin;
        zp->xmax = new_xmax;
        zp->ymin = new_ymin;
        zp->ymax = new_ymax;
        zp->dragging = false;
    }
}

int main() {
    constexpr size_t WIDTH = 800;
    constexpr size_t HEIGHT = 800;

    ZoomParams zp{-2.0, 1.0, -1.5, 1.5};
    namedWindow("Mandelbrot Set", WINDOW_AUTOSIZE);
    setMouseCallback("Mandelbrot Set", on_mouse, &zp);

    Mandelbrot::MandelbrotSet mandelbrot(WIDTH, HEIGHT);

    mandelbrot.setXRange(zp.xmin, zp.xmax).setYRange(zp.ymin, zp.ymax);

    const auto image = mandelbrot.generateImage();

    imshow("Mandelbrot Set", image);
    waitKey(0);
    return 0;
}
