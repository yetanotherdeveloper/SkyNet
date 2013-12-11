cmake_minimum_version(2.8)
project(SkyNet)

set(SOURCE source/main.cpp)

find_package(OpenCL)

add_executable(SkyNet ${SOURCE})
