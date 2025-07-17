// mono_vo/keyframe.h
#pragma once

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
  KeyFrame(long id, const cv::Affine3d & pose) : id_(id), pose_wc_(pose) {}

  // A constructor to create a KeyFrame from a temporary Frame object
  KeyFrame(const Frame & frame)
  {
    id_ = next_keyframe_id_++;
    pose_wc_ = frame.get_pose();
    observations_ = frame.get_observations();
  }

  long get_id() const { return id_; }

  void set_observations(const std::vector<Observation> & observations)
  {
    observations_ = observations;
  }

  const std::vector<Observation> & get_observations() const { return observations_; }

  const cv::Affine3d & get_pose() const { return pose_wc_; }

  const std::vector<cv::Point2f> get_observed_points() const
  {
    std::vector<cv::Point2f> points;
    points.reserve(observations_.size());
    for (const auto & obs : observations_) {
      points.push_back(obs.pt2d);
    }
    return points;
  }

private:
  static long next_keyframe_id_;
  long id_;

  cv::Affine3d pose_wc_;  // Pose of the camera in the world (T_wc)
  std::vector<Observation> observations_;
};

long KeyFrame::next_keyframe_id_ = 0;

}  // namespace mono_vo