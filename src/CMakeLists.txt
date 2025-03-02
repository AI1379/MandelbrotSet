set(MANDELBROT_SET_LIBRARY_SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/MandelbrotSet.cpp
)

add_library(MandelbrotSetLibrary ${MANDELBROT_SET_LIBRARY_SOURCE})

target_include_directories(MandelbrotSetLibrary PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_executable(MandelbrotSet ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(MandelbrotSet MandelbrotSetLibrary)
