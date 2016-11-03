cmake_minimum_required(VERSION 2.8)
project(nn)
set(SOURCE nn.cpp ../../protocol.cpp )
if(${CMAKE_SYSTEM_NAME} MATCHES Linux)
include_directories(../../os/linux)
list(APPEND SOURCE ../../os/linux/os_inc.cpp)
elseif(${CMAKE_SYSTEM_NAME} MATCHES Windows)
include_directories(../../os/windows)
list(APPEND SOURCE ../../os/windows/os_inc.cpp)
endif()

include_directories(./ ../../)
add_definitions(-std=c++11)

add_library(nn SHARED ${SOURCE})

# Unit Tests creation
add_executable(test_nn ${CMAKE_CURRENT_SOURCE_DIR}/test_nn.cpp)

target_link_libraries(test_nn gtest nn dl)
add_test(NAME test_nn COMMAND ./test_nn ) #TODO add working directory

install(TARGETS nn DESTINATION share/skynet/modules)
