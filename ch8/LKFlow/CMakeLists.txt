cmake_minimum_required(VERSION 2.6)
project(lkflow)


set(CMAKE_BUILD_TYPE Release )
set(CMAKE_CXX_FLAGS "-std=c++11 -o3")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(lkflow useLK.cpp)

target_link_libraries(lkflow ${OpenCV_LIBS})