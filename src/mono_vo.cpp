#include "mono_vo/mono_vo.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include <functional>
#include <sophus/se3.hpp>

#include "mono_vo/utils.hpp"

namespace mono_vo
{

MonoVO::MonoVO(const rclcpp::NodeOptions & options)
: Node("mono_vo", options),
  map_(std::make_shared<Map>(this->get_logger().get_child("map"))),
  feature_processor_(
    std::make_shared<FeatureProcessor>(1000, this->get_logger().get_child("feature_processor"))),
  initializer_(map_, feature_processor_, this->get_logger().get_child("initializer")),
  tracker_(map_, feature_processor_, this->get_logger().get_child("tracker"))
{
  this->setup();
}

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

  odometry_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
  RCLCPP_INFO(this->get_logger(), "Publishing to '%s'", odometry_pub_->get_topic_name());

  pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/pointcloud", 10);
  RCLCPP_INFO(this->get_logger(), "Publishing to '%s'", pointcloud_pub_->get_topic_name());

  path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/path", 10);
  RCLCPP_INFO(this->get_logger(), "Publishing to '%s'", path_pub_->get_topic_name());

  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  auto initializer_param_h = RosParameterHandler(this, "initializer");
  initializer_.configure_parameters(initializer_param_h);

  auto tracker_param_h = RosParameterHandler(this, "tracker");
  tracker_.configure_parameters(tracker_param_h);

  RCLCPP_INFO(this->get_logger(), "mono_vo node initialized");
}

void MonoVO::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  RCLCPP_DEBUG(this->get_logger(), "Image message received at ts: '%d'", msg->header.stamp.sec);

  if (!K_.has_value()) {
    RCLCPP_WARN(this->get_logger(), "Waiting for camera info to be published");
    return;
  }

  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  Frame frame{cv_ptr->image};

  if (!initializer_.is_initalized()) {
    std::optional<Frame> ref_frame = initializer_.try_initializing(frame, K_.value());
    if (ref_frame.has_value()) {
      tracker_.update(ref_frame.value(), K_.value(), d_.value());
      RCLCPP_INFO(this->get_logger(), "Initialized");
    }
    return;
  }

  std::optional<Sophus::SE3d> pose_wc = tracker_.update(frame, K_.value(), d_.value());

  if (tracker_.get_state() == TrackerState::LOST) {
    RCLCPP_INFO(this->get_logger(), "Tracker Lost");
    // TODO (Myron): Add resetting logic
    return;
  }

  std_msgs::msg::Header header;
  header.stamp = msg->header.stamp;
  header.frame_id = "odom";
  if (pose_wc.has_value()) {
    nav_msgs::msg::Odometry odometry_msg =
      utils::se3d_to_odometry_msg(pose_wc.value(), header, "camera");
    odometry_pub_->publish(odometry_msg);

    geometry_msgs::msg::TransformStamped transform_msg =
      utils::se3d_to_transform_stamped_msg(pose_wc.value(), header, "camera");
    tf_broadcaster_->sendTransform(transform_msg);

    path_msg_.header = header;
    geometry_msgs::msg::PoseStamped current_pose_stamped;
    current_pose_stamped.pose = odometry_msg.pose.pose;
    current_pose_stamped.header = header;
    path_msg_.poses.push_back(current_pose_stamped);
    path_pub_->publish(path_msg_);
  }

  // publish pointcloud from map
  std::vector<cv::Point3f> points = map_->get_landmark_points();
  sensor_msgs::msg::PointCloud2 pointcloud_msg = utils::points3d_to_pointcloud_msg(points, header);
  pointcloud_pub_->publish(pointcloud_msg);
}

void MonoVO::camera_info_callback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg)
{
  RCLCPP_DEBUG(
    this->get_logger(), "Camera info message received at ts: '%d'", msg->header.stamp.sec);
  if (K_.has_value() && d_.has_value()) return;

  K_ = cv::Mat(3, 3, CV_64F, const_cast<double *>(msg->k.data())).clone();
  d_ = cv::Mat(1, 5, CV_64F, const_cast<double *>(msg->d.data())).clone();
}

}  // namespace mono_vo

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(mono_vo::MonoVO)