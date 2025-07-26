#pragma once

#include <opencv2/opencv.hpp>

#include "mono_vo/frame.hpp"

namespace mono_vo
{
namespace utils
{
/**
 * @brief Draws two frames side by side with matched keypoints.
 *
 * @param frame1 First frame.
 * @param frame2 Second frame.
 * @param draw_keypoints Whether to draw circles around keypoints. Default is true.
 * @param point_radius Radius of circles around keypoints. Default is 4.
 * @param line_thickness Thickness of lines connecting matched keypoints. Default is 1.
 *
 * @return A combined image with the two frames side by side.
 */
cv::Mat draw_matched_frames(
  const Frame & frame1, const Frame & frame2, bool draw_keypoints = true, int point_radius = 4,
  int line_thickness = 1)
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
}  // namespace utils
}  // namespace mono_vo