find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)

if (ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    add_definitions(-DENABLE_OPENMP)
endif ()

set(MANDELBROT_SET_LIBRARY_SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/MandelbrotSet.cpp
)

if (ENABLE_CUDA)
    list(APPEND MANDELBROT_SET_LIBRARY_SOURCE
            ${CMAKE_CURRENT_SOURCE_DIR}/MandelbrotSetCuda.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/ExtendedDouble.cu)
    add_definitions(-DENABLE_CUDA)
endif ()

message(STATUS "Library sources: ${MANDELBROT_SET_LIBRARY_SOURCE}")

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(MandelbrotSetLibrary ${MANDELBROT_SET_LIBRARY_SOURCE})

if (ENABLE_CUDA)
    # For Cuda. From https://github.com/robertmaynard/code-samples/blob/master/posts/cmake/CMakeLists.txt
    target_compile_features(MandelbrotSetLibrary PUBLIC cxx_std_23)
    set_target_properties(MandelbrotSetLibrary
            PROPERTIES CUDA_SEPARABLE_COMPILATION ON
    )
endif ()

target_include_directories(MandelbrotSetLibrary PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(MandelbrotSetLibrary PUBLIC ${OpenCV_LIBS})

if (ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(MandelbrotSetLibrary PUBLIC OpenMP::OpenMP_CXX)
endif ()

add_executable(MandelbrotSet ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(MandelbrotSet PUBLIC MandelbrotSetLibrary)
