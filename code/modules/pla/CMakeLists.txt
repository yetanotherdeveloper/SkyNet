cmake_minimum_required(VERSION 2.8)
project(pla)
set(SOURCE pla.cpp ../../protocol.cpp )
if(${CMAKE_SYSTEM_NAME} MATCHES Linux)
include_directories(../../os/linux)
list(APPEND SOURCE ../../os/linux/os_inc.cpp)
elseif(${CMAKE_SYSTEM_NAME} MATCHES Windows)
include_directories(../../os/windows)
list(APPEND SOURCE ../../os/windows/os_inc.cpp)
endif()

include_directories(./ ../../)
add_definitions(-std=c++11)

add_library(pla SHARED ${SOURCE})

install(TARGETS pla DESTINATION share/skynet/modules)
