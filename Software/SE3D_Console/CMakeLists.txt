cmake_minimum_required(VERSION 2.8)

# External CMake Files Path
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ${CMAKE_MODULE_PATH})

# Project Name
project(SE3D_Console)

# Generate .exe Program
add_executable(${PROJECT_NAME} main.cpp recorder.h recorder.cpp reconstructor.h reconstructor.cpp utility.h utility.cpp)

# Find Packages
find_package(PCL 1.8 REQUIRED)
find_package(OpenCV 2 REQUIRED)
find_package(KinectSDK2 REQUIRED)

if (PCL_FOUND AND KinectSDK2_FOUND AND OpenCV_FOUND)
    # Preprocessor Definitions
    add_definitions( ${PCL_DEFINITIONS} )

    # Include Directories
    include_directories( ${PCL_INCLUDE_DIRS} )
    include_directories( ${OpenCV_INCLUDE_DIRS} )
    include_directories( ${KinectSDK2_INCLUDE_DIRS} )


    # Library Directories
    link_directories(${PCL_LIBRARY_DIRS})
    link_directories(${KinectSDK2_LIBRARY_DIRS})

    # Dependencies
    target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} )
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
    target_link_libraries(${PROJECT_NAME} ${KinectSDK2_LIBRARIES})
endif()
