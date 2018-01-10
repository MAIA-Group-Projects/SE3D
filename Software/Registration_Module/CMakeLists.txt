cmake_minimum_required(VERSION 2.8)

project( model_reconstruction )

find_package( OpenCV REQUIRED )
find_package(PCL 1.2 REQUIRED)
find_package(Qt5Widgets)

include_directories( ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} )

link_directories(${PCL_LIBRARY_DIRS} ${OpenCV_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})

add_executable( ${PROJECT_NAME} main.cpp commonfunc.h commonfunc.cpp)

target_link_libraries( ${PROJECT_NAME} ${PCL_LIBRARIES} ${OpenCV_LIBS} )

qt5_use_modules(${PROJECT_NAME} Widgets)
