cmake_minimum_required(VERSION 2.8)
project(skynet)
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

set(SOURCE main.cpp protocol.cpp skynet.cpp tests/randomPointsClassification.cpp  tests/mnistClassification.cpp)
include_directories(./)
include_directories(./../)  # external headers (opencl) lies here

if(${CMAKE_SYSTEM_NAME} MATCHES Linux)
include_directories(./os/linux)
list(APPEND SOURCE ./os/linux/os_inc.cpp)
elseif(${CMAKE_SYSTEM_NAME} MATCHES Windows)
include_directories(./os/windows)
list(APPEND SOURCE ./os/windows/os_inc.cpp)
endif()

find_package(OpenMP)
if(OPENMP_FOUND)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
fatal("OpenMP support missing in compiler")
endif()

find_package(ImageMagick COMPONENTS Magick++)

if(ImageMagick_Magick++_FOUND)
message("INCLUDES: ${ImageMagick_INCLUDE_DIRS}")
message("LIBS: ${ImageMagick_LIBRARIES}")
else()
message(fatal "No Magic++ of ImageMagic found")
endif()

include_directories(${ImageMagick_INCLUDE_DIRS})




include_directories("/usr/include")
include_directories("/usr/lib/gcc/x86_64-redhat-linux/4.8.3/include")


add_definitions(-std=c++11)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)



add_definitions(-std=c++11)
getsubdirlist(subdirs_list ${CMAKE_CURRENT_SOURCE_DIR})
foreach(subdir ${subdirs_list})
    #message("adding subdir: ${subdir}")
    add_subdirectory(${subdir}) 
endforeach(subdir)

add_executable(skynet ${SOURCE})
target_link_libraries(skynet dl ${ImageMagick_LIBRARIES})

install(TARGETS skynet DESTINATION bin)

set(CPACK_STRIP_FILES TRUE)
set(CPACK_GENERATOR "RPM")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Jacek Czaja")
include(CPack)
