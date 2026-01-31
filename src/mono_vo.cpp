#include "mono_vo/mono_vo.hpp"

#include <cv_bridge/cv_bridge.h>
#include <functional>
#include <chrono>

#include "mono_vo/utils.hpp"

namespace mono_vo
{

MonoVO::MonoVO(const rclcpp::NodeOptions & options)
: Node("mono_vo", options),
  map_(std::make_shared<Map>(this->get_logger().get_child("map"))),
  feature_processor_(
    std::make_shared<FeatureProcessor>(1000, this->get_logger().get_child("feature_processor"))),
  initializer_(map_, feature_processor_, this->get_logger().get_child("initializer")),
  tracker_(map_, feature_processor_, this->get_logger().get_child("tracker")),
  last_valid_pose_stamp_(0, 0, RCL_ROS_TIME)
{
  this->setup();
}

void MonoVO::setup()
{
  // Declare and get frame ID parameters
  this->declare_parameter<std::string>("odom_frame_id", "odom");
  this->declare_parameter<std::string>("odom_child_frame_id", "camera");
  odom_frame_id_ = this->get_parameter("odom_frame_id").as_string();
  odom_child_frame_id_ = this->get_parameter("odom_child_frame_id").as_string();
  RCLCPP_INFO(this->get_logger(), "Using odom_frame_id: '%s', odom_child_frame_id: '%s'",
    odom_frame_id_.c_str(), odom_child_frame_id_.c_str());

  // Declare and get odom publishing parameters
  this->declare_parameter<double>("odom_publish_rate", 30.0);
  this->declare_parameter<double>("position_covariance_growth_rate", 0.1);
  odom_publish_rate_ = this->get_parameter("odom_publish_rate").as_double();
  position_covariance_growth_rate_ = this->get_parameter("position_covariance_growth_rate").as_double();
  RCLCPP_INFO(this->get_logger(), "Odom publish rate: %.1f Hz, covariance growth rate: %.3f m^2/s",
    odom_publish_rate_, position_covariance_growth_rate_);

  // Use SensorDataQoS (best effort) to match video_receiver's publisher QoS
  auto sensor_qos = rclcpp::SensorDataQoS();

  image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "/camera/image_rect", sensor_qos,
    [this](const sensor_msgs::msg::Image::ConstSharedPtr & msg) { image_callback(msg); });

  RCLCPP_INFO(this->get_logger(), "Subscribed to '%s' (QoS: best_effort)", image_sub_->get_topic_name());

  camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    "/camera/camera_info", sensor_qos, [this](const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg) {
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

  // Create odom polling timer
  odom_timer_ = this->create_wall_timer(
    std::chrono::duration<double>(1.0 / odom_publish_rate_),
    std::bind(&MonoVO::odom_timer_callback, this));
  RCLCPP_INFO(this->get_logger(), "Odom polling timer started at %.1f Hz", odom_publish_rate_);

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
      // Set initial pose (identity at origin)
      std::lock_guard<std::mutex> lock(pose_mutex_);
      last_pose_ = cv::Affine3d::Identity();
      last_valid_pose_stamp_ = msg->header.stamp;
      tracking_valid_ = true;
    }
    return;
  }

  std::optional<cv::Affine3d> pose_wc = tracker_.update(frame, K_.value(), d_.value());

  // Update pose state (thread-safe)
  {
    std::lock_guard<std::mutex> lock(pose_mutex_);
    if (tracker_.get_state() == TrackerState::LOST) {
      RCLCPP_DEBUG(this->get_logger(), "Tracker Lost - continuing with last known pose");
      tracking_valid_ = false;
      // Don't return - let timer continue publishing last pose
    } else if (pose_wc.has_value()) {
      last_pose_ = pose_wc.value();
      last_valid_pose_stamp_ = msg->header.stamp;
      tracking_valid_ = true;
    }
  }

  // Update path (only when tracking is valid)
  if (tracking_valid_ && last_pose_.has_value()) {
    std_msgs::msg::Header header;
    header.stamp = msg->header.stamp;
    header.frame_id = odom_frame_id_;

    nav_msgs::msg::Odometry odom_msg =
      utils::affine3d_to_odometry_msg(last_pose_.value(), header, odom_child_frame_id_);

    path_msg_.header = header;
    geometry_msgs::msg::PoseStamped current_pose_stamped;
    current_pose_stamped.pose = odom_msg.pose.pose;
    current_pose_stamped.header = header;
    path_msg_.poses.push_back(current_pose_stamped);
    path_pub_->publish(path_msg_);

    // publish pointcloud from map
    std::vector<cv::Point3f> points = map_->get_landmark_points();
    sensor_msgs::msg::PointCloud2 pointcloud_msg = utils::points3d_to_pointcloud_msg(points, header);
    pointcloud_pub_->publish(pointcloud_msg);
  }
}

void MonoVO::odom_timer_callback()
{
  std::lock_guard<std::mutex> lock(pose_mutex_);

  if (!last_pose_.has_value()) {
    // Not initialized yet, don't publish
    return;
  }

  publish_odom(this->now(), tracking_valid_);
}

void MonoVO::publish_odom(const rclcpp::Time & stamp, bool is_valid)
{
  std_msgs::msg::Header header;
  header.stamp = stamp;
  header.frame_id = odom_frame_id_;

  nav_msgs::msg::Odometry odom_msg =
    utils::affine3d_to_odometry_msg(last_pose_.value(), header, odom_child_frame_id_);

  // When tracking is lost, increase covariance based on time elapsed
  if (!is_valid) {
    double time_since_valid = (stamp - last_valid_pose_stamp_).seconds();
    double cov_increase = position_covariance_growth_rate_ * time_since_valid;

    // Increase position covariance (diagonal elements 0, 7, 14)
    odom_msg.pose.covariance[0] += cov_increase;   // x
    odom_msg.pose.covariance[7] += cov_increase;   // y
    odom_msg.pose.covariance[14] += cov_increase;  // z

    // Also increase rotation covariance slightly (diagonal elements 21, 28, 35)
    odom_msg.pose.covariance[21] += cov_increase * 0.1;  // roll
    odom_msg.pose.covariance[28] += cov_increase * 0.1;  // pitch
    odom_msg.pose.covariance[35] += cov_increase * 0.1;  // yaw
  }

  odometry_pub_->publish(odom_msg);

  // Also publish TF
  geometry_msgs::msg::TransformStamped tf_msg =
    utils::affine3d_to_transform_stamped_msg(last_pose_.value(), header, odom_child_frame_id_);
  tf_broadcaster_->sendTransform(tf_msg);
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