#pragma once

#include <algorithm>
#include <opencv2/opencv.hpp>
#include <vector>

#include "mono_vo/frame.hpp"
#include "mono_vo/landmark.hpp"

namespace mono_vo
{

class KeyFrame
{
public:
  using Ptr = std::shared_ptr<KeyFrame>;
  explicit KeyFrame(const cv::Affine3d & pose);

  bool isAffine3dDefault(const cv::Affine3d & pose, double eps = 1e-9);

  // A constructor to create a KeyFrame from a temporary Frame object
  explicit KeyFrame(const Frame & frame);

  void add_observation(
    const cv::KeyPoint & keypoint, const cv::Mat & descriptor, long landmark_id = -1);

  std::vector<cv::Point2f> get_points_2d(
    ObservationFilter filter_type = ObservationFilter::ALL) const;

  std::vector<cv::Point2f> get_points_2d_for_landmarks(
    const std::vector<long> & landmark_ids) const;

  const Observation * get_observation_for_landmark(long landmark_id) const;

  std::vector<cv::KeyPoint> get_keypoints() const;

  cv::Mat get_descriptors() const;

  std::vector<Observation> get_observations(
    ObservationFilter filter_type = ObservationFilter::ALL) const;

  void clear_observations();

  long id;
  cv::Affine3d
    pose_wc;  // Pose of the camera in the world (T_wc), takes point in camera to the world frame
  std::vector<Observation> observations;

private:
  static long next_id_;
  std::unordered_map<long, Observation *> landmark_id_to_observation;
};

}  // namespace mono_vo