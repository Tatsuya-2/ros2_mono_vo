#pragma once

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include "mono_vo/frame.hpp"

namespace mono_vo
{
namespace utils
{

/**
 * @brief Draws matched points between two images.
 *
 * This function takes two input images and their corresponding matched points,
 * draws circles around the matched points, and connects them with lines. The
 * images are concatenated side by side, and matches are drawn with random colors
 * for better visualization.
 *
 * @param img1 First input image.
 * @param img2 Second input image.
 * @param points1 Matched points in the first image.
 * @param points2 Matched points in the second image.
 * @param point_radius Radius of the circles drawn around matched points. Default is 4.
 * @param line_thickness Thickness of the lines connecting matched points. Default is 1.
 *
 * @return A single image with both input images side by side and matched points visualized.
 */
cv::Mat draw_matched_points(
  const cv::Mat & img1, const cv::Mat & img2, const std::vector<cv::Point2f> & points1,
  const std::vector<cv::Point2f> & points2, int point_radius = 4, int line_thickness = 1);

nav_msgs::msg::Odometry affine3d_to_odometry_msg(
  const cv::Affine3d & pose_wc, const std_msgs::msg::Header & header,
  const std::string & child_frame_id);

geometry_msgs::msg::TransformStamped affine3d_to_transform_stamped_msg(
  const cv::Affine3d & pose_wc, const std_msgs::msg::Header & header,
  const std::string & child_frame_id);

/**
 * @brief Converts a vector of 3D points from an OpenCV-convention map frame to a
 *        sensor_msgs::msg::PointCloud2 message in a ROS-convention map frame.
 *
 * This function is crucial for visualizing map points from a typical VO/SLAM system in RViz.
 * It assumes the input points are in a frame where X is right, Y is down, and Z is forward.
 * It transforms them to the standard ROS frame where X is forward, Y is left, and Z is up.
 *
 * @param points The vector of 3D points (cv::Point3f) from the VO map.
 * @param header The ROS message header. The header.frame_id should be set to your
 *               fixed world frame (e.g., "map" or "odom").
 * @return sensor_msgs::msg::PointCloud2 The resulting point cloud message, ready to be published.
 */
sensor_msgs::msg::PointCloud2 points3d_to_pointcloud_msg(
  const std::vector<cv::Point3f> & points, const std_msgs::msg::Header & header);

/**
 * @brief Computes the reprojection error between a 3D world point and its projection in an image.
 *
 * This function projects a 3D world point onto an image using the provided projection matrix P and
 * computes the Euclidean distance between the projected 2D point and the observed 2D point.
 *
 * @param p3d_world The 3D point in the world coordinate system.
 * @param p2d_observed The observed 2D point in the image.
 * @param P The 3x4 projection matrix.
 *
 * @return The reprojection error between the projected and observed 2D points.
 */
double compute_reprojection_error(
  const cv::Point3f & p3d_world, const cv::Point2f & p2d_observed, const cv::Mat & P);

}  // namespace utils
}  // namespace mono_vo