cmake_minimum_required(VERSION 2.6)
project(ch7)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3.1 REQUIRED)
find_package( G2O REQUIRED )
find_package(CSparse REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${G2O_INCLUDE_DIR}
  ${CSPARSE_INCLUDE_DIR}
  "/use/include/eigen3/"
)


add_executable(feature_extraction feature_extraction.cpp)
target_link_libraries(feature_extraction ${OpenCV_LIBS})

add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})







