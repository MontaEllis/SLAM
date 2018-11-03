cmake_minimum_required(VERSION 2.6)
project(ceres_curve_fitting)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIR})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_DIRS})


add_executable(ceres_curve_fitting main.cpp)
target_link_libraries(ceres_curve_fitting ${CERES_LIBRARIES} ${OpenCV_LIBS})

