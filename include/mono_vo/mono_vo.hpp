#pragma once

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <memory>
#include <nav_msgs/msg/path.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>
#include <vector>

#include "mono_vo/feature_processor.hpp"
#include "mono_vo/initializer.hpp"
#include "mono_vo/map.hpp"
#include "mono_vo/tracker.hpp"

namespace mono_vo
{

class MonoVO : public rclcpp::Node
{
public:
  explicit MonoVO(const rclcpp::NodeOptions & options);

private:
  void setup();

  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);

  void camera_info_callback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg);

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;

  std::shared_ptr<Map> map_;
  FeatureProcessor::Ptr feature_processor_;
  Initializer initializer_;
  Tracker tracker_;

  std::optional<cv::Mat> K_;
  std::optional<cv::Mat> d_;

  nav_msgs::msg::Path path_msg_;
};

}  // namespace mono_vo
