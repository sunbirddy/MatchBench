cmake_minimum_required(VERSION 2.8)
project( FeatureMatchers )

SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE__BUILD_TYPE "RELEASE")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# pencv_matcher: sift surf orb kaze akaze brisk
add_executable( opencv_matchers ${PROJECT_SOURCE_DIR}/src/opencv_matchers.cpp )
target_link_libraries( opencv_matchers ${OpenCV_LIBS} )

# gms_matcher
add_executable( gms_matcher ${PROJECT_SOURCE_DIR}/src/gms_matcher.cpp )
target_link_libraries( gms_matcher ${OpenCV_LIBS} )

# usac_matcher
add_executable( usac_matcher ${PROJECT_SOURCE_DIR}/src/usac_matcher.cpp )
target_link_libraries( usac_matcher ${OpenCV_LIBS} )

# usac_matcher_time
add_executable( usac_matcher_time ${PROJECT_SOURCE_DIR}/src/usac_matcher_time.cpp )
target_link_libraries( usac_matcher_time ${OpenCV_LIBS} )


# time_reporter
add_executable( time_reporter ${PROJECT_SOURCE_DIR}/src/time_reporter.cpp )
target_link_libraries( time_reporter ${OpenCV_LIBS} )

# time_reporter_gpu
add_executable( time_reporter_gpu ${PROJECT_SOURCE_DIR}/src/time_reporter_gpu.cpp )
target_link_libraries( time_reporter_gpu ${OpenCV_LIBS} )
