cmake_minimum_required(VERSION 2.8)
project( Tools )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( ConstructDatasetfromTUM ConstructDatasetfromTUM.cpp )
target_link_libraries( ConstructDatasetfromTUM ${OpenCV_LIBS} )

add_executable( ConstructDatasetfromKITTI ConstructDatasetfromKITTI.cpp )
target_link_libraries( ConstructDatasetfromKITTI ${OpenCV_LIBS} )

add_executable( DownsampleTUM DownsampleTUM.cpp )
target_link_libraries( DownsampleTUM ${OpenCV_LIBS} )

