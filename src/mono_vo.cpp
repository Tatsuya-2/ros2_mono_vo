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

  camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    "/camera/camera_info", 10, [this](const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg) {
      camera_info_callback(msg);
    });
  RCLCPP_INFO(this->get_logger(), "Subscribed to '%s'", camera_info_sub_->get_topic_name());

  pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/camera/pose", 10);
  RCLCPP_INFO(this->get_logger(), "Publishing to '%s'", pose_pub_->get_topic_name());

  RCLCPP_INFO(this->get_logger(), "Node initialized");
}

void MonoVO::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  RCLCPP_DEBUG(this->get_logger(), "Image message received at ts: '%d'", msg->header.stamp.sec);

  if (!K_.has_value()) {
    RCLCPP_INFO(this->get_logger(), "Waiting for camera info");
    return;
  }

  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }
  cv::Mat image = cv_ptr->image;

  initializer_.try_initializing(image, K_.value());
}

void MonoVO::camera_info_callback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg)
{
  RCLCPP_DEBUG(
    this->get_logger(), "Camera info message received at ts: '%d'", msg->header.stamp.sec);
  if (K_.has_value()) return;
  K_ = cv::Matx33d(
    msg->k[0], msg->k[1], msg->k[2], msg->k[3], msg->k[4], msg->k[5], msg->k[6], msg->k[7],
    msg->k[8]);
}

}  // namespace mono_vo

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(mono_vo::MonoVO)