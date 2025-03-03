set(MANDELBROT_SET_LIBRARY_SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/MandelbrotSet.cpp
)

find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)

find_package(OpenMP)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(MandelbrotSetLibrary ${MANDELBROT_SET_LIBRARY_SOURCE})
target_include_directories(MandelbrotSetLibrary PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(MandelbrotSetLibrary PUBLIC ${OpenCV_LIBS})

if (OpenMP_CXX_FOUND)
    target_link_libraries(MandelbrotSetLibrary PUBLIC OpenMP::OpenMP_CXX)
endif ()

add_executable(MandelbrotSet ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(MandelbrotSet PUBLIC MandelbrotSetLibrary)
