# 프로젝트 루트 CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(calibration)

# OpenCV 설정
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

# 서브 디렉토리 추가
add_subdirectory(tools)
add_subdirectory(single_camera)
add_subdirectory(stereo_camera)
