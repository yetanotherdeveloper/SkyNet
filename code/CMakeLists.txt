cmake_minimum_required(VERSION 2.8)
project(SkyNet)
macro(getsubdirlist result curdir)
    file(GLOB items RELATIVE ${curdir} ${curdir}/modules/*)
    set(dirlist "")
    foreach(item ${items})
        if(IS_DIRECTORY ${curdir}/${item})
            set(dirlist ${dirlist} ${item})
        endif()
    endforeach(item)
    set(${result} ${dirlist} )
endmacro(getsubdirlist)

if(${CMAKE_BUILD_TYPE} MATCHES Debug)
add_definitions(-DDEBUG)
endif()

set(SOURCE main.cpp protocol.cpp skynet.cpp tests/randomPointsClassification.cpp)
include_directories(./)
include_directories(./../)  # external headers (opencl) lies here

if(${CMAKE_SYSTEM_NAME} MATCHES Linux)
include_directories(./os/linux)
list(APPEND SOURCE ./os/linux/os_inc.cpp)
elseif(${CMAKE_SYSTEM_NAME} MATCHES Windows)
include_directories(./os/windows)
list(APPEND SOURCE ./os/windows/os_inc.cpp)
endif()

add_definitions(-std=c++11)
getsubdirlist(subdirs_list ${CMAKE_CURRENT_SOURCE_DIR})
foreach(subdir ${subdirs_list})
    #message("adding subdir: ${subdir}")
    add_subdirectory(${subdir}) 
endforeach(subdir)

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
find_library(OPENCL_LIB OpenCL PATHS lib/opencl)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
find_library(OPENCL_LIB OpenCL PATHS /usr/lib)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")

include(./cmake/copy_clang_complete.cmake )

add_executable(SkyNet ${SOURCE})
target_link_libraries(SkyNet dl ${OPENCL_LIB})

install(TARGETS SkyNet DESTINATION bin)

set(CPACK_STRIP_FILES TRUE)
set(CPACK_GENERATOR "RPM")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Jacek Czaja")
include(CPack)
