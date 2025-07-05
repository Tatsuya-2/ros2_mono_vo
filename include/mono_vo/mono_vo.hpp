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

/**
 * @brief MonoVo class
 */
class MonoVO : public rclcpp::Node
{
public:
  /**
   * @brief Constructor
   *
   * @param options node options
   */
  explicit MonoVO(const rclcpp::NodeOptions & options);

private:
  /**
   * @brief Sets up subscribers, publishers, etc. to configure the node
   */
  void setup();

  /**
   * @brief Processes messages received by a subscriber
   *
   * @param msg message
   */
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);

private:
  /**
   * @brief Subscriber
   */
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

  /**
   * @brief Publisher
   */
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
};

}  // namespace mono_vo
