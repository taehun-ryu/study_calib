# single_camera 실행 파일 생성
# src/main.cpp을 빌드 대상으로 설정
add_executable(single_camera src/main.cpp)

# 새로 추가된 헤더 파일이 있는 include 디렉토리를 추가
target_include_directories(single_camera PRIVATE include)

# tools 라이브러리 및 OpenCV 라이브러리를 링크
target_link_libraries(single_camera PRIVATE tools ${OpenCV_LIBS} Eigen3::Eigen Ceres::ceres)

# 빌드 설정 출력
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")