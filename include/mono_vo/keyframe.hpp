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
  }

  void add_observation(
    const cv::KeyPoint & keypoint, const cv::Mat & descriptor, long landmark_id = -1)
  {
    observations.emplace_back(keypoint, descriptor, landmark_id);
  }

  std::vector<cv::Point2f> get_points_2d() const
  {
    std::vector<cv::Point2f> points_2d;
    points_2d.reserve(observations.size());
    for (const auto & obs : observations) {
      points_2d.push_back(obs.keypoint.pt);
    }
    return points_2d;
  }

  std::vector<cv::Point2f> get_points_2d_for_landmarks(const std::vector<long> & landmark_ids) const
  {
    std::vector<cv::Point2f> points_2d;
    points_2d.reserve(landmark_ids.size());
    for (const auto landmark_id : landmark_ids) {
      if (const auto obs = std::find_if(
            observations.begin(), observations.end(),
            [landmark_id](const Observation & obs) { return obs.landmark_id == landmark_id; });
          obs != observations.end()) {
        points_2d.push_back(obs->keypoint.pt);
      }
    }
    points_2d.shrink_to_fit();
    return points_2d;
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

  long id;
  cv::Affine3d
    pose_wc;  // Pose of the camera in the world (T_wc), takes point in camera to the world frame
  std::vector<Observation> observations;

private:
  static long next_id_;
};

long KeyFrame::next_id_ = 0;

}  // namespace mono_vo