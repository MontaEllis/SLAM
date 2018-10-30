cmake_minimum_required(VERSION 2.6)
project(ch3_useeigen)

add_executable(ch3_useeigen main.cpp)
# h
include_directories("/usr/include/eigen3")

install(TARGETS ch3_useeigen RUNTIME DESTINATION bin)
