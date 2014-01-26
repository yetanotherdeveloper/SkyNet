cmake_minimum_required(VERSION 2.8)
project(pla)
set(SOURCE pla.cpp ../../protocol.cpp)
include_directories(./ ../../)
add_definitions(-std=c++11)

find_package(OpenCL)
add_library(pla SHARED ${SOURCE})
