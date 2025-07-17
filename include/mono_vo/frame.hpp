// mono_vo/frame.h
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

#include "mono_vo/landmark.hpp"

namespace mono_vo
{
struct Observation
{
  long landmark_id;
  cv::Point2f pt2d;
  cv::Mat descriptor;
};

// Temporary frame to store data before it is added to the map
class Frame
{
public:
  Frame(const cv::Mat & image, const cv::Affine3d & pose = cv::Affine3d())
  : id_(next_frame_id_++), image_(image.clone()), pose_wc_(pose)
  {
  }

  void set_observations(const std::vector<Observation> & observations)
  {
    observations_ = observations;
  }

  // Helper to get 2D points for optical flow
  std::vector<cv::Point2f> get_observed_points() const
  {
    std::vector<cv::Point2f> points;
    points.reserve(observations_.size());
    for (const auto & obs : observations_) {
      points.push_back(obs.pt2d);
    }
    return points;
  }

  const std::vector<Observation> & get_observations() const { return observations_; }

  const cv::Affine3d & get_pose() const { return pose_wc_; }

private:
  static long next_frame_id_;

  long id_;
  cv::Mat image_;
  cv::Affine3d pose_wc_;  // Pose of the camera in the world (T_wc)
  std::vector<Observation> observations_;
};

long Frame::next_frame_id_ = 0;
}  // namespace mono_vo