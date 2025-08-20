#include "mono_vo/utils.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <cstring>
#include <sophus/se3.hpp>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "std_msgs/msg/header.hpp"

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

nav_msgs::msg::Odometry se3d_to_odometry_msg(
  const Sophus::SE3d & pose_wc, const std_msgs::msg::Header & header,
  const std::string & child_frame_id)
{
  // Coordinate frame transformation matrix
  const static Eigen::Matrix3d cv_to_ros_rotation =
    (Eigen::Matrix3d() << 0, 0, 1, -1, 0, 0, 0, -1, 0).finished();

  // Apply coordinate transformation to the SE3 directly
  Sophus::SE3d pose_ros = Sophus::SE3d(cv_to_ros_rotation, Eigen::Vector3d::Zero()) * pose_wc *
                          Sophus::SE3d(cv_to_ros_rotation.transpose(), Eigen::Vector3d::Zero());

  // Extract components from transformed pose
  const Eigen::Vector3d & t_ros = pose_ros.translation();
  const Eigen::Quaterniond & q_eigen = pose_ros.unit_quaternion();

  // Assemble the Odometry message
  nav_msgs::msg::Odometry odom_msg;
  odom_msg.header = header;
  odom_msg.child_frame_id = child_frame_id;
  odom_msg.pose.pose.position.x = t_ros.x();
  odom_msg.pose.pose.position.y = t_ros.y();
  odom_msg.pose.pose.position.z = t_ros.z();
  odom_msg.pose.pose.orientation.x = q_eigen.x();
  odom_msg.pose.pose.orientation.y = q_eigen.y();
  odom_msg.pose.pose.orientation.z = q_eigen.z();
  odom_msg.pose.pose.orientation.w = q_eigen.w();

  // Set covariance values
  odom_msg.pose.covariance[0] = 0.1;    // x
  odom_msg.pose.covariance[7] = 0.1;    // y
  odom_msg.pose.covariance[14] = 0.1;   // z
  odom_msg.pose.covariance[21] = 0.05;  // rotation x
  odom_msg.pose.covariance[28] = 0.05;  // rotation y
  odom_msg.pose.covariance[35] = 0.05;  // rotation z

  odom_msg.twist.covariance[0] = 1e-3;
  odom_msg.twist.covariance[7] = 1e-3;
  odom_msg.twist.covariance[35] = 1e-3;

  return odom_msg;
}

geometry_msgs::msg::TransformStamped se3d_to_transform_stamped_msg(
  const Sophus::SE3d & pose_wc, const std_msgs::msg::Header & header,
  const std::string & child_frame_id)
{
  // Coordinate frame transformation matrix
  const static Eigen::Matrix3d cv_to_ros_rotation =
    (Eigen::Matrix3d() << 0, 0, 1, -1, 0, 0, 0, -1, 0).finished();

  // Apply coordinate transformation to the SE3 directly
  Sophus::SE3d pose_ros = Sophus::SE3d(cv_to_ros_rotation, Eigen::Vector3d::Zero()) * pose_wc *
                          Sophus::SE3d(cv_to_ros_rotation.transpose(), Eigen::Vector3d::Zero());

  // Extract components from transformed pose
  const Eigen::Vector3d & t_ros = pose_ros.translation();
  const Eigen::Quaterniond & q_eigen = pose_ros.unit_quaternion();

  // Assemble the TransformStamped message
  geometry_msgs::msg::TransformStamped t_stamped;
  t_stamped.header = header;
  t_stamped.child_frame_id = child_frame_id;
  t_stamped.transform.translation.x = t_ros.x();
  t_stamped.transform.translation.y = t_ros.y();
  t_stamped.transform.translation.z = t_ros.z();
  t_stamped.transform.rotation.x = q_eigen.x();
  t_stamped.transform.rotation.y = q_eigen.y();
  t_stamped.transform.rotation.z = q_eigen.z();
  t_stamped.transform.rotation.w = q_eigen.w();

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

cv::Affine3d se3d_to_affine3d(const Sophus::SE3d & se3)
{
  const Eigen::Matrix4d & T = se3.matrix();

  // OpenCV uses row-major, Eigen uses column-major
  cv::Matx44d cv_matrix(
    T(0, 0), T(0, 1), T(0, 2), T(0, 3), T(1, 0), T(1, 1), T(1, 2), T(1, 3), T(2, 0), T(2, 1),
    T(2, 2), T(2, 3), T(3, 0), T(3, 1), T(3, 2), T(3, 3));

  return cv::Affine3d(cv_matrix);
}

Sophus::SE3d affine3d_to_se3d(const cv::Affine3d & affine)
{
  // Get direct access to the matrix data
  const cv::Matx44d & cv_mat = affine.matrix;

  // Construct Eigen matrix directly from OpenCV data
  Eigen::Matrix4d T;
  T << cv_mat(0, 0), cv_mat(0, 1), cv_mat(0, 2), cv_mat(0, 3), cv_mat(1, 0), cv_mat(1, 1),
    cv_mat(1, 2), cv_mat(1, 3), cv_mat(2, 0), cv_mat(2, 1), cv_mat(2, 2), cv_mat(2, 3),
    cv_mat(3, 0), cv_mat(3, 1), cv_mat(3, 2), cv_mat(3, 3);

  return Sophus::SE3d(T);
}

}  // namespace utils
}  // namespace mono_vo