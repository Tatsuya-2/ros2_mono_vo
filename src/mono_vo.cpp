#include <functional>
#include <mono_vo/mono_vo.hpp>
#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(mono_vo::MonoVO)

namespace mono_vo
{

MonoVO::MonoVO(const rclcpp::NodeOptions & options) : Node("mono_vo", options) { this->setup(); }

void MonoVO::setup()
{
  // subscriber for handling incoming messages
  subscriber_ = this->create_subscription<std_msgs::msg::Int32>(
    "~/input", 10, std::bind(&MonoVO::topicCallback, this, std::placeholders::_1));
  RCLCPP_INFO(this->get_logger(), "Subscribed to '%s'", subscriber_->get_topic_name());

  // publisher for publishing outgoing messages
  publisher_ = this->create_publisher<std_msgs::msg::Int32>("~/output", 10);
  RCLCPP_INFO(this->get_logger(), "Publishing to '%s'", publisher_->get_topic_name());
}

void MonoVO::topicCallback(const std_msgs::msg::Int32::ConstSharedPtr & msg)
{
  RCLCPP_INFO(this->get_logger(), "Message received: '%d'", msg->data);

  // publish message
  std_msgs::msg::Int32::UniquePtr out_msg = std::make_unique<std_msgs::msg::Int32>();
  out_msg->data = msg->data;
  publisher_->publish(std::move(out_msg));
  RCLCPP_INFO(this->get_logger(), "Message published: '%d'", out_msg->data);
}

}  // namespace mono_vo
