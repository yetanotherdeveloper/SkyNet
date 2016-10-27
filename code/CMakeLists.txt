cmake_minimum_required(VERSION 2.8)
project(skynet)
enable_testing()
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

# Copy profile.stp and put absolute path to binary and nn lib
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/profile.stp DESTINATION ${CMAKE_BINARY_DIR})
file(READ ${CMAKE_BINARY_DIR}/profile.stp PROFILE_STP_STRING)
string(REPLACE "[DIRNAME_OF_SKYNET_BINARY]" "\"${CMAKE_BINARY_DIR}/skynet\"" PROFILE_STP_STRING ${PROFILE_STP_STRING})
string(REPLACE "[DIRNAME_OF_NN_SKYNET_MODULE]" "\"${CMAKE_BINARY_DIR}/modules/nn/libnn.so\"" PROFILE_STP_STRING ${PROFILE_STP_STRING})
file(WRITE ${CMAKE_BINARY_DIR}/profile.stp ${PROFILE_STP_STRING})

install(TARGETS skynet DESTINATION bin)


# Unit Tests creation
add_executable(test_commandline tests/test_commandline.cpp skynet.cpp os/linux/os_inc.cpp)

target_link_libraries(test_commandline gtest dl)
add_test(NAME test_commandline COMMAND ./test_commandline ) #TODO add working directory

set(CPACK_STRIP_FILES TRUE)
set(CPACK_GENERATOR "RPM")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Jacek Czaja")
include(CPack)
