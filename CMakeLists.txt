cmake_minimum_required(VERSION 3.8)
project(mono_vo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

set(TARGET_NAME mono_vo_component)

add_library(${TARGET_NAME} SHARED src/mono_vo.cpp src/feature_extractor.cpp 
src/frame.cpp src/initializer.cpp src/keyframe.cpp src/landmark.cpp src/map.cpp 
src/match_data.cpp src/tracker.cpp src/utils.cpp)

target_precompile_headers(${TARGET_NAME} PRIVATE
  <rclcpp/rclcpp.hpp>
  <sensor_msgs/msg/image.hpp>
  <sensor_msgs/msg/camera_info.hpp>
  <sensor_msgs/msg/point_cloud2.hpp>
  <nav_msgs/msg/path.hpp>
  <geometry_msgs/msg/pose_stamped.hpp>
  <opencv2/opencv.hpp>
)

target_compile_features(${TARGET_NAME} PUBLIC c_std_99 cxx_std_17)

ament_target_dependencies(${TARGET_NAME}
  rclcpp
  rclcpp_components
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2
  cv_bridge
)

target_include_directories(${TARGET_NAME} 
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  SYSTEM PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${TARGET_NAME}
 ${OpenCV_LIBS}
)

rclcpp_components_register_node(${TARGET_NAME}
  PLUGIN "mono_vo::MonoVO"
  EXECUTABLE mono_vo  
)

install(TARGETS ${TARGET_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

install(DIRECTORY rviz
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()
