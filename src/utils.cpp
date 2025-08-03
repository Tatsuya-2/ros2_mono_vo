#include "mono_vo/utils.hpp"

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

#include <cmath>
#include <cstring>

namespace mono_vo
{
namespace utils
{

cv::Mat draw_matched_frames(
  const Frame & frame1, const Frame & frame2, bool draw_keypoints, int point_radius,
  int line_thickness)
{
  // Ensure both images have the same number of channels
  cv::Mat img1_color, img2_color;

  if (frame1.image.channels() == 1) {
    cv::cvtColor(frame1.image, img1_color, cv::COLOR_GRAY2BGR);
  } else {
    img1_color = frame1.image.clone();
  }

  if (frame2.image.channels() == 1) {
    cv::cvtColor(frame2.image, img2_color, cv::COLOR_GRAY2BGR);
  } else {
    img2_color = frame2.image.clone();
  }

  // Create a combined image (side by side)
  int combined_width = img1_color.cols + img2_color.cols;
  int combined_height = std::max(img1_color.rows, img2_color.rows);

  cv::Mat combined_img(combined_height, combined_width, CV_8UC3, cv::Scalar(0, 0, 0));

  // Copy images to combined image
  cv::Mat left_roi = combined_img(cv::Rect(0, 0, img1_color.cols, img1_color.rows));
  img1_color.copyTo(left_roi);

  cv::Mat right_roi = combined_img(cv::Rect(img1_color.cols, 0, img2_color.cols, img2_color.rows));
  img2_color.copyTo(right_roi);

  std::vector<cv::KeyPoint> keypoints1 = frame1.get_keypoints();
  std::vector<cv::KeyPoint> keypoints2 = frame2.get_keypoints();

  // Draw keypoints and matches
  size_t num_matches = std::min(keypoints1.size(), keypoints2.size());

  // Predefined colors for better visualization
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

  for (size_t i = 0; i < num_matches; ++i) {
    // Get keypoint positions
    cv::Point2f pt1 = keypoints1[i].pt;
    cv::Point2f pt2 = keypoints2[i].pt;

    // Adjust pt2 coordinates for the right image position
    pt2.x += img1_color.cols;

    // Get color (cycle through predefined colors)
    cv::Scalar color = colors[i % colors.size()];

    if (draw_keypoints) {
      // Draw circles around keypoints
      cv::circle(combined_img, pt1, point_radius, color, -1);
      cv::circle(combined_img, pt2, point_radius, color, -1);

      // Draw keypoint centers in white for better visibility
      cv::circle(combined_img, pt1, 1, cv::Scalar(255, 255, 255), -1);
      cv::circle(combined_img, pt2, 1, cv::Scalar(255, 255, 255), -1);
    }

    // Draw line connecting the matches
    cv::line(combined_img, pt1, pt2, color, line_thickness, cv::LINE_AA);
  }

  return combined_img;
}

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

geometry_msgs::msg::PoseStamped affine3d_to_pose_stamped_msg(
  const cv::Affine3d & affine_pose, const std_msgs::msg::Header & header)
{
  geometry_msgs::msg::PoseStamped pose_stamped;

  // Set header
  pose_stamped.header = header;

  // Extract translation from affine transform
  cv::Vec3d translation = affine_pose.translation();
  pose_stamped.pose.position.x = translation[0];
  pose_stamped.pose.position.y = translation[1];
  pose_stamped.pose.position.z = translation[2];

  // Extract rotation matrix from affine transform
  // Extract rotation matrix
  const cv::Matx33d & rot_cv = affine_pose.rotation();

  // Step 1: Define OpenCV → ROS coordinate transform (90° around X-axis)
  // This maps:
  //   OpenCV: X=right, Y=down,  Z=forward
  //   ROS:    X=forward, Y=left, Z=up
  cv::Matx33d T_OC2ROS(0, -1, 0, 0, 0, -1, 1, 0, 0);

  // Step 2: Apply rotation transform to convert from OpenCV to ROS frame
  cv::Matx33d rot_ros = rot_cv * T_OC2ROS;  // Rotate the orientation

  tf2::Matrix3x3 tf_rotation(
    rot_ros(0, 0), rot_ros(0, 1), rot_ros(0, 2), rot_ros(1, 0), rot_ros(1, 1), rot_ros(1, 2),
    rot_ros(2, 0), rot_ros(2, 1), rot_ros(2, 2));

  // Convert to quaternion
  tf2::Quaternion quaternion;
  tf_rotation.getRotation(quaternion);

  // Set quaternion in pose message
  pose_stamped.pose.orientation.x = quaternion.x();
  pose_stamped.pose.orientation.y = quaternion.y();
  pose_stamped.pose.orientation.z = quaternion.z();
  pose_stamped.pose.orientation.w = quaternion.w();

  return pose_stamped;
}

sensor_msgs::msg::PointCloud2 points3d_to_pointcloud_msg(
  const std::vector<cv::Point3f> & points, const std_msgs::msg::Header & header)
{
  sensor_msgs::msg::PointCloud2 cloud_msg;

  // Set the header
  cloud_msg.header = header;

  // Set the cloud properties
  cloud_msg.height = 1;  // Unordered point cloud
  cloud_msg.width = points.size();
  cloud_msg.is_dense = true;  // Assume no invalid points (NaNs, infs)

  // Define the fields (x, y, z)
  cloud_msg.fields.resize(3);

  cloud_msg.fields[0].name = "x";
  cloud_msg.fields[0].offset = 0;
  cloud_msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud_msg.fields[0].count = 1;

  cloud_msg.fields[1].name = "y";
  cloud_msg.fields[1].offset = 4;  // sizeof(float)
  cloud_msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud_msg.fields[1].count = 1;

  cloud_msg.fields[2].name = "z";
  cloud_msg.fields[2].offset = 8;  // 2 * sizeof(float)
  cloud_msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud_msg.fields[2].count = 1;

  // Set the point step and row step
  // point_step is the size of a single point in bytes
  cloud_msg.point_step = 3 * sizeof(float);  // 3 fields (x, y, z) * 4 bytes/field
  // row_step is the total size of a row in bytes
  cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;

  // Set the data
  // The total size of the data buffer
  size_t data_size = cloud_msg.row_step * cloud_msg.height;
  cloud_msg.data.resize(data_size);

  // This is the core of the conversion.
  // We copy the raw memory from the vector of cv::Point3f to the data buffer
  // of the PointCloud2 message. This is efficient because cv::Point3f is a
  // Plain Old Data (POD) type with 3 consecutive floats, which matches the
  // desired layout.
  memcpy(cloud_msg.data.data(), points.data(), data_size);

  // Check for non-finite points (NaN or Inf) and update is_dense flag
  for (const auto & point : points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      cloud_msg.is_dense = false;
      break;
    }
  }

  // This field is not super important, but we can set it correctly.
  // It's true if the machine is big-endian, false for little-endian.
  // Most modern systems (x86, ARM) are little-endian.
  const int i = 1;
  cloud_msg.is_bigendian = (*(char *)&i == 0);

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