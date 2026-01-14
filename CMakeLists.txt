cmake_minimum_required(VERSION 3.16)
project(DefectDetector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

#add_executable(cv_test src/main.cpp)
add_executable(cv_test
  src/main.cpp
  src/detector.cpp
  )
  
  target_link_libraries(cv_test PRIVATE ${OpenCV_LIBS})
  target_include_directories(cv_test PRIVATE ${OpenCV_INCLUDE_DIRS})
  
  if (MSVC)
  add_compile_options(/utf-8)  
  endif()
  