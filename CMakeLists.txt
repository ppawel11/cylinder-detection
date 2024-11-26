cmake_minimum_required(VERSION 3.22)
project(cylinder_detection)

set(CMAKE_CXX_STANDARD 17)

set(depthai_DIR "/home/pawel/luxonis/depthai-core/build1/")

find_package(depthai CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(VTK 9.1 REQUIRED)
add_executable(cylinder_detection main.cpp)

include_directories(/usr/include/vtk-9.1 ${PCL_INCLUDE_DIRS})
target_link_libraries(cylinder_detection PRIVATE depthai::opencv ${OpenCV_LIBS} Threads::Threads ${PCL_LIBRARIES})
