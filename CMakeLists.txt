cmake_minimum_required(VERSION 3.22)
project(try_to_start)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
find_path(KAFKA_INCLUDE_PATH NAMES "rdkafkacpp.h" HINTS "C:/Program Files (x86)/RdKafka/include/*")
include_directories(${KAFKA_INCLUDE_PATH})
include_directories(opencv_functional)
message(STATUS "includes ${OpenCV_INCLUDE_DIRS} \n${OpenCV_LIBRARIES}")
include_directories(${OpenCV_INCLUDE_DIRS})
#add_subdirectory(opencv_functional)


add_executable(try_to_start main.cpp)
add_executable(client wannabe-client.cpp)
add_executable(server wannabe-server.cpp)
#set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
add_library(img_func opencv_functional/opencv_functional.cpp opencv_functional/opencv_functional.h)
target_link_libraries(img_func ${OpenCV_LIBRARIES})
target_link_libraries(try_to_start ${OpenCV_LIBRARIES} img_func)
find_library(KAFKA_LIBRARY rdkafka++ HINTS "C:/Program Files (x86)/RdKafka/lib")
message(STATUS "includes \n ${KAFKA_INCLUDE_PATH}
\n ${KAFKA_LIBRARY}" )
target_link_libraries(client ${KAFKA_LIBRARY} ${OpenCV_LIBRARIES})
target_link_libraries(server ${OpenCV_LIBRARIES} ${KAFKA_LIBRARY} img_func)



