cmake_minimum_required(VERSION 2.6)
project(imagebasic)
set( CMAKE_CXX_FLAGS "-std=c++11")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(imagebasic main.cpp 123.jpg)
target_link_libraries(imagebasic ${OpenCV_LIBS})

install(TARGETS imagebasic RUNTIME DESTINATION bin)
