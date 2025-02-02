cmake_minimum_required(VERSION 3.10)
project(calibration)

# 전처리기 세팅
option(FIX_INTRINSIC "Fix intrinsic parameters" OFF)

# 인터페이스 라이브러리 생성
add_library(global_defines INTERFACE)
add_library(single_camera_defines INTERFACE)
add_library(stereo_camera_defines INTERFACE)

# 전처리기 - 인터페이스 연결
if(FIX_INTRINSIC)
  target_compile_definitions(stereo_camera_defines INTERFACE FIX_INTRINSIC=1)
else()
  target_compile_definitions(stereo_camera_defines INTERFACE FIX_INTRINSIC=0)
endif()

# OpenCV, Eigen3, Ceres 설정
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

# 서브 디렉토리 추가
add_subdirectory(tools)
add_subdirectory(single_camera)
add_subdirectory(stereo_camera)
