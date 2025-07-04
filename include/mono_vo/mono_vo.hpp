#pragma once

#include <memory>
#include <rclcpp/rclcpp.hpp>
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
  void topicCallback(const std_msgs::msg::Int32::ConstSharedPtr & msg);

private:
  /**
   * @brief Subscriber
   */
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr subscriber_;

  /**
   * @brief Publisher
   */
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr publisher_;
};

}  // namespace mono_vo
