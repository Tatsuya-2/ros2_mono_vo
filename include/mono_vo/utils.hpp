#pragma once

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
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

geometry_msgs::msg::PoseStamped affine3d_to_pose_stamped_msg(
  const cv::Affine3d & affine_pose, const std_msgs::msg::Header & header);

/**
 * @brief Converts a std::vector<cv::Point3f> to a sensor_msgs::msg::PointCloud2.
 *
 * @param points The input vector of 3D points (cv::Point3f).
 * @param header The ROS header to use for the PointCloud2 message (contains frame_id and stamp).
 * @return A sensor_msgs::msg::PointCloud2 message.
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