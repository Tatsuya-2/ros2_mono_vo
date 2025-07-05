#pragma once

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/int32.hpp>
#include <string>
#include <vector>

namespace mono_vo
{

class MonoVO : public rclcpp::Node
{
public:
  explicit MonoVO(const rclcpp::NodeOptions & options);

private:
  void setup();

  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
};

}  // namespace mono_vo
