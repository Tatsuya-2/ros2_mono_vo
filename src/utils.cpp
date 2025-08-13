#include "mono_vo/utils.hpp"

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

#include <cmath>
#include <cstring>

namespace mono_vo
{
namespace utils
{

cv::Mat draw_matched_points(
  const cv::Mat & img1, const cv::Mat & img2, const std::vector<cv::Point2f> & points1,
  const std::vector<cv::Point2f> & points2, int point_radius, int line_thickness)
{
  // Ensure both images are grayscale
  cv::Mat gray1, gray2;
  if (img1.channels() == 3) {
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
  } else {
    gray1 = img1.clone();
  }

  if (img2.channels() == 3) {
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
  } else {
    gray2 = img2.clone();
  }

  // Create output image: concatenate images side by side
  int height = std::max(gray1.rows, gray2.rows);
  int width = gray1.cols + gray2.cols;

  cv::Mat result = cv::Mat::zeros(height, width, CV_8UC3);

  // Convert grayscale to BGR for colored output
  cv::Mat color1, color2;
  cv::cvtColor(gray1, color1, cv::COLOR_GRAY2BGR);
  cv::cvtColor(gray2, color2, cv::COLOR_GRAY2BGR);

  // Copy images to result
  color1.copyTo(result(cv::Rect(0, 0, gray1.cols, gray1.rows)));
  color2.copyTo(result(cv::Rect(gray1.cols, 0, gray2.cols, gray2.rows)));

  // Check if point vectors have same size
  size_t num_matches = std::min(points1.size(), points2.size());

  // Color palette
  std::vector<cv::Scalar> colors = {
    cv::Scalar(0, 0, 255),    // Red
    cv::Scalar(0, 255, 0),    // Green
    cv::Scalar(255, 0, 0),    // Blue
    cv::Scalar(255, 255, 0),  // Cyan
    cv::Scalar(255, 0, 255),  // Magenta
    cv::Scalar(0, 255, 255),  // Yellow
    cv::Scalar(128, 0, 128),  // Purple
    cv::Scalar(0, 128, 128)   // Teal
  };

  // Draw matches
  for (size_t i = 0; i < num_matches; ++i) {
    // Generate random color for this match
    const cv::Scalar & color = colors[i % colors.size()];

    // Points in original coordinates
    cv::Point2f pt1 = points1[i];
    cv::Point2f pt2 = points2[i];

    // Adjust pt2 coordinates for concatenated image
    cv::Point2f pt2_adjusted(pt2.x + gray1.cols, pt2.y);

    // Draw circles at matched points
    cv::circle(result, pt1, point_radius, color, -1);
    cv::circle(result, pt2_adjusted, point_radius, color, -1);

    // Draw line connecting the points
    cv::line(result, pt1, pt2_adjusted, color, line_thickness);
  }

  return result;
}

nav_msgs::msg::Odometry affine3d_to_odometry_msg(
  const cv::Affine3d & pose_wc, const std_msgs::msg::Header & header,
  const std::string & child_frame_id)
{
  // Constant transform from OpenCV camera frame to ROS standard frame (REP 103)
  // ROS X-fwd, Y-left, Z-up <=> OpenCV Z-fwd, X-right, Y-down
  // ROS X = OpenCV Z
  // ROS Y = -OpenCV X
  // ROS Z = -OpenCV Y
  const static cv::Matx33d cv_to_ros_rotation(0, 0, 1, -1, 0, 0, 0, -1, 0);

  // Extract and convert pose
  cv::Matx33d R_cv = pose_wc.rotation();
  cv::Vec3d t_cv = pose_wc.translation();

  // Rotate the rotation matrix (conjugation)
  cv::Matx33d R_ros = cv_to_ros_rotation * R_cv * cv_to_ros_rotation.t();
  // Rotate the translation vector
  cv::Vec3d t_ros = cv_to_ros_rotation * t_cv;

  // Convert rotation matrix to quaternion
  tf2::Matrix3x3 tf_rotation(
    R_ros(0, 0), R_ros(0, 1), R_ros(0, 2), R_ros(1, 0), R_ros(1, 1), R_ros(1, 2), R_ros(2, 0),
    R_ros(2, 1), R_ros(2, 2));

  // Convert to quaternion
  tf2::Quaternion q_ros;
  tf_rotation.getRotation(q_ros);
  q_ros.normalize();

  // Assemble the Odometry message
  nav_msgs::msg::Odometry odom_msg;
  odom_msg.header = header;
  odom_msg.child_frame_id = child_frame_id;

  odom_msg.pose.pose.position.x = t_ros[0];
  odom_msg.pose.pose.position.y = t_ros[1];
  odom_msg.pose.pose.position.z = t_ros[2];
  odom_msg.pose.pose.orientation.x = q_ros.x();
  odom_msg.pose.pose.orientation.y = q_ros.y();
  odom_msg.pose.pose.orientation.z = q_ros.z();
  odom_msg.pose.pose.orientation.w = q_ros.w();

  // TODO (myron): Needs tuning based on observations
  odom_msg.pose.covariance[0] = 0.1;    // x
  odom_msg.pose.covariance[7] = 0.1;    // y
  odom_msg.pose.covariance[14] = 0.1;   // z
  odom_msg.pose.covariance[21] = 0.05;  // rotation x
  odom_msg.pose.covariance[28] = 0.05;  // rotation y
  odom_msg.pose.covariance[35] = 0.05;  // rotation z

  // Twist is not provided by a single pose, so leave it zero
  // and indicate high uncertainty in covariance
  odom_msg.twist.covariance[0] = 1e-3;
  odom_msg.twist.covariance[7] = 1e-3;
  odom_msg.twist.covariance[35] = 1e-3;

  return odom_msg;
}

geometry_msgs::msg::TransformStamped affine3d_to_transform_stamped_msg(
  const cv::Affine3d & pose_wc, const std_msgs::msg::Header & header,
  const std::string & child_frame_id)
{
  const static cv::Matx33d cv_to_ros_rotation(0, 0, 1, -1, 0, 0, 0, -1, 0);

  // Extract and convert pose
  cv::Matx33d R_cv = pose_wc.rotation();
  cv::Vec3d t_cv = pose_wc.translation();

  cv::Vec3d t_ros = cv_to_ros_rotation * t_cv;
  cv::Matx33d R_ros = cv_to_ros_rotation * R_cv * cv_to_ros_rotation.t();

  // Convert rotation matrix to quaternion
  tf2::Matrix3x3 tf_rotation(
    R_ros(0, 0), R_ros(0, 1), R_ros(0, 2), R_ros(1, 0), R_ros(1, 1), R_ros(1, 2), R_ros(2, 0),
    R_ros(2, 1), R_ros(2, 2));

  // Convert to quaternion
  tf2::Quaternion q_ros;
  tf_rotation.getRotation(q_ros);
  q_ros.normalize();

  // Assemble the TransformStamped message
  geometry_msgs::msg::TransformStamped t_stamped;
  t_stamped.header = header;
  t_stamped.child_frame_id = child_frame_id;

  t_stamped.transform.translation.x = t_ros[0];
  t_stamped.transform.translation.y = t_ros[1];
  t_stamped.transform.translation.z = t_ros[2];
  t_stamped.transform.rotation.x = q_ros.x();
  t_stamped.transform.rotation.y = q_ros.y();
  t_stamped.transform.rotation.z = q_ros.z();
  t_stamped.transform.rotation.w = q_ros.w();

  return t_stamped;
}

sensor_msgs::msg::PointCloud2 points3d_to_pointcloud_msg(
  const std::vector<cv::Point3f> & points, const std_msgs::msg::Header & header)
{
  sensor_msgs::msg::PointCloud2 cloud_msg;

  // Set the header
  cloud_msg.header = header;

  // Set basic cloud properties
  cloud_msg.height = 1;  // Unordered point cloud
  cloud_msg.width = points.size();
  cloud_msg.is_dense = true;  // Assume no NaN or Inf values
  cloud_msg.is_bigendian = false;

  // Define the fields (x, y, z)
  sensor_msgs::msg::PointField field;
  field.name = "x";
  field.offset = 0;
  field.datatype = sensor_msgs::msg::PointField::FLOAT32;
  field.count = 1;
  cloud_msg.fields.push_back(field);

  field.name = "y";
  field.offset = 4;  // x is 4 bytes (float32)
  field.datatype = sensor_msgs::msg::PointField::FLOAT32;
  field.count = 1;
  cloud_msg.fields.push_back(field);

  field.name = "z";
  field.offset = 8;  // y is 4 bytes
  field.datatype = sensor_msgs::msg::PointField::FLOAT32;
  field.count = 1;
  cloud_msg.fields.push_back(field);

  // Set the size of a single point and the total data size
  cloud_msg.point_step = 3 * sizeof(float);  // 12 bytes for x, y, z
  cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;
  cloud_msg.data.resize(cloud_msg.row_step);

  // Fill the data buffer with transformed points
  // We use a pointer to navigate the byte buffer
  uint8_t * data_ptr = cloud_msg.data.data();

  for (const auto & pt_cv : points) {
    // Apply the coordinate frame transformation for each point
    // ROS X = CV Z
    // ROS Y = -CV X
    // ROS Z = -CV Y
    float pt_ros[3];
    pt_ros[0] = pt_cv.z;
    pt_ros[1] = -pt_cv.x;
    pt_ros[2] = -pt_cv.y;

    // Copy the 3 floats (12 bytes) into the buffer
    memcpy(data_ptr, pt_ros, cloud_msg.point_step);
    data_ptr += cloud_msg.point_step;
  }

  return cloud_msg;
}

double compute_reprojection_error(
  const cv::Point3f & p3d_world, const cv::Point2f & p2d_observed, const cv::Mat & P)
{
  // Project the 3D point
  cv::Mat p4d_world = (cv::Mat_<double>(4, 1) << p3d_world.x, p3d_world.y, p3d_world.z, 1.0);
  cv::Mat p3d_projected_h = P * p4d_world;
  cv::Point2f p2d_projected;
  p2d_projected.x = p3d_projected_h.at<double>(0) / p3d_projected_h.at<double>(2);
  p2d_projected.y = p3d_projected_h.at<double>(1) / p3d_projected_h.at<double>(2);

  return cv::norm(p2d_projected - p2d_observed);
}

}  // namespace utils
}  // namespace mono_vo