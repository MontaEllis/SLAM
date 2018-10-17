cmake_minimum_required(VERSION 2.6)
project(usegeometry)

add_executable(usegeometry main.cpp)

install(TARGETS usegeometry RUNTIME DESTINATION bin)
