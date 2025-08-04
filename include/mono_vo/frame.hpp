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

enum class ObservationFilter
{
  ALL,
  WITH_LANDMARKS,
  WITHOUT_LANDMARKS
};

// Temporary frame to store data before it is added to the map
struct Frame
{
public:
  using Ptr = std::shared_ptr<Frame>;
  explicit Frame(const cv::Mat & image);

  void extract_observations(const FeatureExtractor::Ptr extractor);

  void add_observation(const cv::KeyPoint & keypoint, const cv::Mat & descriptor, long landmark_id);

  std::vector<cv::Point2f> get_points_2d(
    ObservationFilter filter_type = ObservationFilter::ALL) const;

  std::vector<long> get_landmark_ids() const;

  std::vector<cv::KeyPoint> get_keypoints() const;

  cv::Mat get_descriptors() const;

  std::vector<Observation> get_observations(
    ObservationFilter filter_type = ObservationFilter::ALL) const;

  void clear_observations();

  long id;
  cv::Mat image;
  cv::Affine3d pose_wc;  // Pose of the camera in the world (T_wc), takes point in camera to world
  std::vector<Observation> observations;
  bool is_tracked =
    false;  // indicates if the frame has passed the tracking stage which invalidates keypoint fields
};

}  // namespace mono_vo