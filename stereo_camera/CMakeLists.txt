# stereo_camera 실행 파일 생성
# src/main.cpp을 빌드 대상으로 설정
add_executable(stereo_camera src/main.cpp)

# 새로 추가된 헤더 파일이 있는 include 디렉토리를 추가
target_include_directories(stereo_camera PRIVATE include ${CMAKE_CURRENT_SOURCE_DIR}/../single_camera/include)

# tools 라이브러리 및 OpenCV 라이브러리를 링크
target_link_libraries(stereo_camera PRIVATE
  global_defines
  stereo_camera_defines
  tools
  ${OpenCV_LIBS}
  Eigen3::Eigen
  Ceres::ceres
)
