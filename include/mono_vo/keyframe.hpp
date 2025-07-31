// mono_vo/keyframe.h
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
  explicit KeyFrame(const cv::Affine3d & pose) : id(next_id_++), pose_wc(pose) {}

  bool isAffine3dDefault(const cv::Affine3d & pose, double eps = 1e-9)
  {
    return cv::norm(pose.matrix - cv::Affine3d().matrix) < eps;
  }
  // A constructor to create a KeyFrame from a temporary Frame object
  explicit KeyFrame(const Frame & frame)
  {
    // assert that the frame is not empty
    assert(!frame.observations.empty());
    assert(!isAffine3dDefault(frame.pose_wc, 1e-9));
    id = next_id_++;
    pose_wc = frame.pose_wc;
    observations = frame.observations;
    landmark_id_to_observation.reserve(observations.size());
    for (auto & obs : observations) {
      if (obs.landmark_id != -1) {
        landmark_id_to_observation[obs.landmark_id] = &obs;
      }
    }
  }

  void add_observation(
    const cv::KeyPoint & keypoint, const cv::Mat & descriptor, long landmark_id = -1)
  {
    observations.emplace_back(keypoint, descriptor, landmark_id);
    if (landmark_id != -1) {
      landmark_id_to_observation[landmark_id] = &observations.back();
    }
  }

  std::vector<cv::Point2f> get_points_2d(
    ObservationFilter filter_type = ObservationFilter::ALL) const
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
          throw std::runtime_error("Invalid filter type in KeyFrame::get_points_2d()");
          break;
      }
    }
    points_2d.shrink_to_fit();
    return points_2d;
  }

  std::vector<cv::Point2f> get_points_2d_for_landmarks(const std::vector<long> & landmark_ids) const
  {
    std::vector<cv::Point2f> points_2d;
    points_2d.reserve(landmark_ids.size());
    for (const auto landmark_id : landmark_ids) {
      if (auto obs = landmark_id_to_observation.at(landmark_id); obs != nullptr) {
        points_2d.push_back(obs->keypoint.pt);
      }
    }
    points_2d.shrink_to_fit();
    return points_2d;
  }

  const Observation * get_observation_for_landmark(long landmark_id) const
  {
    return landmark_id_to_observation.at(landmark_id);
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

  std::vector<Observation> get_observations(
    ObservationFilter filter_type = ObservationFilter::ALL) const
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

  void clear_observations() { observations.clear(); }

  long id;
  cv::Affine3d
    pose_wc;  // Pose of the camera in the world (T_wc), takes point in camera to the world frame
  std::vector<Observation> observations;

private:
  static long next_id_;
  std::unordered_map<long, Observation *> landmark_id_to_observation;
};

long KeyFrame::next_id_ = 0;

}  // namespace mono_vo