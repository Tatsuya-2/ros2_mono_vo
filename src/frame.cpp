#include "mono_vo/frame.hpp"

namespace mono_vo
{

Frame::Frame(const cv::Mat & image) : image(image.clone()) {}

void Frame::extract_features(const FeatureExtractor::Ptr extractor)
{
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  extractor->detect_and_compute(image, keypoints, descriptors);
  observations.reserve(keypoints.size());
  for (size_t i = 0; i < keypoints.size(); i++) {
    add_observation(keypoints[i], descriptors.row(i), -1);
  }
}

void Frame::add_observation(
  const cv::KeyPoint & keypoint, const cv::Mat & descriptor, long landmark_id)
{
  observations.emplace_back(keypoint, descriptor, landmark_id);
}

std::vector<cv::Point2f> Frame::get_points_2d(ObservationFilter filter_type) const
{
  std::vector<cv::Point2f> points_2d;
  points_2d.reserve(observations.size());
  for (const auto & obs : observations) {
    switch (filter_type) {
      case ObservationFilter::WITHOUT_LANDMARKS:
        if (obs.landmark_id == -1) points_2d.push_back(obs.keypoint.pt);
        break;
      case ObservationFilter::WITH_LANDMARKS:
        if (obs.landmark_id != -1) points_2d.push_back(obs.keypoint.pt);
        break;
      case ObservationFilter::ALL:
        points_2d.push_back(obs.keypoint.pt);
        break;

      default:
        throw std::runtime_error("Invalid filter type in Frame::get_points_2d()");
        break;
    }
  }
  return points_2d;
}

std::vector<long> Frame::get_landmark_ids() const
{
  std::vector<long> landmark_ids;
  landmark_ids.reserve(observations.size());
  for (const auto & obs : observations) {
    landmark_ids.push_back(obs.landmark_id);
  }
  return landmark_ids;
}

std::vector<cv::KeyPoint> Frame::get_keypoints() const
{
  std::vector<cv::KeyPoint> keypoints;
  keypoints.reserve(observations.size());
  for (const auto & obs : observations) {
    keypoints.push_back(obs.keypoint);
  }
  return keypoints;
}

cv::Mat Frame::get_descriptors() const
{
  std::vector<cv::Mat> descriptor_rows;
  for (const auto & obs : observations) {
    descriptor_rows.push_back(obs.descriptor);
  }
  cv::Mat descriptors;
  if (!descriptor_rows.empty()) {
    cv::vconcat(descriptor_rows, descriptors);
  }
  return descriptors;
}

void Frame::filter_observations_by_mask(const std::vector<uchar> & inlier_mask)
{
  std::vector<Observation> in_obs;
  in_obs.reserve(inlier_mask.size());
  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    if (inlier_mask[i]) {
      in_obs.push_back(observations[i]);
    }
  }
  observations = in_obs;
}

std::vector<Observation> Frame::get_observations(ObservationFilter filter_type) const
{
  if (filter_type == ObservationFilter::ALL) {
    return observations;
  }
  std::vector<Observation> valid_obs;
  valid_obs.reserve(observations.size());
  for (const auto & obs : observations) {
    switch (filter_type) {
      case ObservationFilter::WITH_LANDMARKS:
        if (obs.landmark_id != -1) valid_obs.push_back(obs);
        break;

      case ObservationFilter::WITHOUT_LANDMARKS:
        if (obs.landmark_id == -1) valid_obs.push_back(obs);
        break;

      default:
        throw std::runtime_error("Invalid filter type for Frame::get_observations()");
        break;
    }
  }
  return valid_obs;
}

void Frame::clear_observations() { observations.clear(); }

}  // namespace mono_vo