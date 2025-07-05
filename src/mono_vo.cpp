#include <cv_bridge/cv_bridge.hpp>
#include <functional>
#include <mono_vo/mono_vo.hpp>

namespace mono_vo
{

MonoVO::MonoVO(const rclcpp::NodeOptions & options) : Node("mono_vo", options) { this->setup(); }

void MonoVO::setup()
{
  image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "/camera/image_rect", 10,
    [this](const sensor_msgs::msg::Image::ConstSharedPtr & msg) { image_callback(msg); });

  RCLCPP_INFO(this->get_logger(), "Subscribed to '%s'", image_sub_->get_topic_name());

  pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/camera/pose", 10);
  RCLCPP_INFO(this->get_logger(), "Publishing to '%s'", pose_pub_->get_topic_name());
}

void MonoVO::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  RCLCPP_INFO(this->get_logger(), "Image message received at ts: '%d'", msg->header.stamp.sec);

  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }
  cv::Mat image = cv_ptr->image;

  initializer_.update(image);
}

}  // namespace mono_vo

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(mono_vo::MonoVO)