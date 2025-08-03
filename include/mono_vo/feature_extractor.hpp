#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include <rclcpp/logging.hpp>

namespace mono_vo
{
class FeatureExtractor
{
public:
  using Ptr = std::shared_ptr<FeatureExtractor>;

  FeatureExtractor(
    int num_features = 1000, rclcpp::Logger logger = rclcpp::get_logger("FeatureExtractor"));

  std::vector<cv::KeyPoint> detect(const cv::Mat & image) const;

  void detect_and_compute(
    const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints, cv::Mat & descriptors) const;

  /**
   * Find matches between two sets of descriptors.
   *
   * @param descriptors1 Descriptors from the first image.
   * @param descriptors2 Descriptors from the second image.
   * @param lowes_distance_ratio The distance ratio between the best and second best match.
   * @return A vector of DMatch objects, where each match is represented by a DMatch object.
   */
  std::vector<cv::DMatch> find_matches(
    const cv::Mat & descriptors1, const cv::Mat & descriptors2, double lowes_distance_ratio) const;

private:
  cv::Ptr<cv::ORB> detector_;
  cv::BFMatcher matcher_;
  rclcpp::Logger logger_;
};
}  // namespace mono_vo