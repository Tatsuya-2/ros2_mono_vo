// mono_vo/frame.h
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

#include "mono_vo/feature_extractor.hpp"
#include "mono_vo/landmark.hpp"

namespace mono_vo
{
class Observation
{
public:
  explicit Observation(
    const cv::KeyPoint & keypoint, const cv::Mat & descriptor, long landmark_id = -1)
  : keypoint(keypoint), descriptor(descriptor), landmark_id(landmark_id)
  {
  }
  cv::KeyPoint keypoint;
  cv::Mat descriptor;
  long landmark_id;
};
// Temporary frame to store data before it is added to the map
struct Frame
{
public:
  using Ptr = std::shared_ptr<Frame>;
  explicit Frame(const cv::Mat & image) : image(image.clone()) {}

  void extract_features(const FeatureExtractor::Ptr extractor)
  {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    extractor->detect_and_compute(image, keypoints, descriptors);
    observations.reserve(keypoints.size());
    for (size_t i = 0; i < keypoints.size(); i++) {
      add_observation(keypoints[i], descriptors.row(i), -1);
    }
  }

  void add_observation(const cv::KeyPoint & keypoint, const cv::Mat & descriptor, long landmark_id)
  {
    observations.emplace_back(keypoint, descriptor, landmark_id);
  }

  std::vector<cv::Point2f> get_points_2d(bool with_valid_landmarks = false) const
  {
    std::vector<cv::Point2f> points_2d;
    points_2d.reserve(observations.size());
    for (const auto & obs : observations) {
      if (with_valid_landmarks && obs.landmark_id == -1) {
        continue;
      }
      points_2d.push_back(obs.keypoint.pt);
    }
    points_2d.shrink_to_fit();
    return points_2d;
  }

  std::vector<long> get_landmark_ids() const
  {
    std::vector<long> landmark_ids;
    landmark_ids.reserve(observations.size());
    for (const auto & obs : observations) {
      landmark_ids.push_back(obs.landmark_id);
    }
    return landmark_ids;
  }

  std::vector<cv::KeyPoint> get_keypoints() const
  {
    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(observations.size());
    for (const auto & obs : observations) {
      keypoints.push_back(obs.keypoint);
    }
    return keypoints;
  }

  cv::Mat get_descriptors() const
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

  void filter_observations_by_mask(const std::vector<uchar> & inlier_mask)
  {
    std::vector<Observation> in_obs;
    in_obs.reserve(inlier_mask.size());
    for (size_t i = 0; i < inlier_mask.size(); ++i) {
      if (inlier_mask[i]) {
        in_obs.push_back(observations[i]);
      }
    }
    in_obs.shrink_to_fit();
    observations = in_obs;
  }

  std::vector<Observation> get_observations(bool with_valid_landmarks = false) const
  {
    if (!with_valid_landmarks) {
      return observations;
    }
    std::vector<Observation> valid_obs;
    valid_obs.reserve(observations.size());
    for (const auto & obs : observations) {
      if (obs.landmark_id != -1) {
        valid_obs.push_back(obs);
      }
    }
    return valid_obs;
  }

  long id;
  cv::Mat image;
  cv::Affine3d pose_wc;  // Pose of the camera in the world (T_wc), takes point in camera to world
  std::vector<Observation> observations;
  bool is_tracked =
    false;  // indicates if the frame has passed the tracking stage which invalidates keypoint fields
};

}  // namespace mono_vo