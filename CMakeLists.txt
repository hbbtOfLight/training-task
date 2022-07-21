cmake_minimum_required(VERSION 3.22)
project(try_to_start)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
message(STATUS "includes ${OpenCV_INCLUDE_DIRS} \n${OpenCV_LIBRARIES}")
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(try_to_start main.cpp)
#set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
target_link_libraries(try_to_start ${OpenCV_LIBRARIES})

