cmake_minimum_required(VERSION 2.8)
project(svm)
set(SOURCE svm.cpp ../../protocol.cpp)

include_directories(./ ../../)
add_definitions(-std=c++11)

find_package(OpenCL)
add_library(svm SHARED ${SOURCE})
