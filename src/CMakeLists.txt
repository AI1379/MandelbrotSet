find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)

set(MANDELBROT_SET_SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/MandelbrotSet.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

set(MANDELBROT_SET_DEPENDENCIES
        ${OpenCV_LIBS}
)

if (ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    add_definitions(-DENABLE_OPENMP)
    list(APPEND MANDELBROT_SET_DEPENDENCIES OpenMP::OpenMP_CXX)
endif ()

if (ENABLE_MPFR)
    find_package(MPFR REQUIRED)
    list(APPEND MANDELBROT_SET_SOURCE
            ${CMAKE_CURRENT_SOURCE_DIR}/MandelbrotSetMPFR.cpp)
    message(STATUS "MPFR include dir: ${MPFR_INCLUDE_DIR}")
    add_definitions(-DENABLE_MPFR)
    list(APPEND MANDELBROT_SET_DEPENDENCIES MPFR::MPFR)
endif ()

if (ENABLE_CUDA)
    list(APPEND MANDELBROT_SET_SOURCE
            ${CMAKE_CURRENT_SOURCE_DIR}/MandelbrotSetCuda.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/ExtendedDouble.cu)
    add_definitions(-DENABLE_CUDA)
endif ()

if (ENABLE_STDEXEC)
    find_package(STDEXEC REQUIRED)
    add_definitions(-DENABLE_STDEXEC)
    list(APPEND MANDELBROT_SET_DEPENDENCIES STDEXEC::stdexec)
endif ()

message(STATUS "Project sources: ${MANDELBROT_SET_SOURCE}")
message(STATUS "Project dependencies: ${MANDELBROT_SET_DEPENDENCIES}")

include_directories(${PROJECT_SOURCE_DIR}/src)

add_executable(MandelbrotSet ${MANDELBROT_SET_SOURCE})

if (ENABLE_CUDA)
    # For Cuda. From https://github.com/robertmaynard/code-samples/blob/master/posts/cmake/CMakeLists.txt
    target_compile_features(MandelbrotSet PUBLIC cxx_std_23)
    set_target_properties(MandelbrotSet
            PROPERTIES CUDA_SEPARABLE_COMPILATION ON
    )
endif ()

target_link_libraries(MandelbrotSet PRIVATE ${MANDELBROT_SET_DEPENDENCIES})