#include "mono_vo/feature_extractor.hpp"

namespace mono_vo
{
FeatureExtractor::FeatureExtractor(int num_features, rclcpp::Logger logger)
: detector_(cv::ORB::create(num_features)),
  matcher_(cv::BFMatcher(cv::NORM_HAMMING)),
  logger_(logger)
{
}

std::vector<cv::KeyPoint> FeatureExtractor::detect(const cv::Mat & image) const
{
  std::vector<cv::KeyPoint> keypoints;
  detector_->detect(image, keypoints);
  return keypoints;
}

void FeatureExtractor::detect_and_compute(
  const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints, cv::Mat & descriptors) const
{
  detector_->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
}

std::vector<cv::DMatch> FeatureExtractor::find_matches(
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

}  // namespace mono_vo