cmake_minimum_required(VERSION 2.6)
project(test)

add_executable(test main.cpp)

install(TARGETS test RUNTIME DESTINATION bin)
