#cmake -DOpenCV_DIR=C:\Ceemple\OpenCV4VS\lib

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(DETECTOR)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake/modules/")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(OpenCV_LIBS ${OpenCV_LIBS} ${OpenCV_LIBRARIES})

add_executable(ObjectDetector main.cpp
							MonoContourObjectDetector.cpp
							MiltiContourObjectDetector.cpp
							ObjectDetector.cpp
							ObjectDetectorFactory.cpp
							Utility.cpp)

target_link_libraries(ObjectDetector ${OpenCV_LIBS})