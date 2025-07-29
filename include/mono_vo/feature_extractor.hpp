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
    int num_features = 1000, rclcpp::Logger logger = rclcpp::get_logger("FeatureExtractor"))
  : detector_(cv::ORB::create(num_features)),
    matcher_(cv::BFMatcher(cv::NORM_HAMMING)),
    logger_(logger)
  {
  }

  std::vector<cv::KeyPoint> detect(const cv::Mat & image) const
  {
    std::vector<cv::KeyPoint> keypoints;
    detector_->detect(image, keypoints);
    return keypoints;
  }

  void detect_and_compute(
    const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints, cv::Mat & descriptors) const
  {
    detector_->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
  }

  /**
   * Find matches between two sets of descriptors.
   *
   * @param descriptors1 Descriptors from the first image.
   * @param descriptors2 Descriptors from the second image.
   * @param lowes_distance_ratio The distance ratio between the best and second best match.
   * @return A vector of DMatch objects, where each match is represented by a DMatch object.
   */
  std::vector<cv::DMatch> find_matches(
    const cv::Mat & descriptors1, const cv::Mat & descriptors2, double lowes_distance_ratio) const
  {
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_.knnMatch(descriptors1, descriptors2, knn_matches, 2);
    RCLCPP_INFO(logger_, "total matches: %ld", knn_matches.size());

    std::vector<cv::DMatch> good_matches;
    for (auto & match : knn_matches) {
      if (match.size() == 2 && match[0].distance < lowes_distance_ratio * match[1].distance) {
        good_matches.push_back(match[0]);
      }
    }

    RCLCPP_INFO(logger_, "good matches: %ld", good_matches.size());
    return good_matches;
  }

private:
  cv::Ptr<cv::ORB> detector_;
  cv::BFMatcher matcher_;
  rclcpp::Logger logger_;
};
}  // namespace mono_vo