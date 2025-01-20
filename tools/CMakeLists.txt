# tools 라이브러리 생성 (헤더 파일만 포함)
add_library(tools STATIC CharucoBoard.hpp)

# 링커 언어를 C++로 지정 -> hpp로만 되어있기 때문에 사용
set_target_properties(tools PROPERTIES LINKER_LANGUAGE CXX)

# 헤더 파일 경로 설정
target_include_directories(tools PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
