#pragma once

#include <opencv2/opencv.hpp>

namespace mono_vo
{
struct Landmark
{
  cv::Point3f pose;
  cv::Mat descriptor;
};

struct TrackedKP
{
  long landmark_id;
  cv::Point2f pt2d;
};

class Map
{
public:
  Map() = default;

  void add_new_landmark(
    const cv::Point3f & point, const cv::Mat & descriptor, const cv::Point2f & keypoint)
  {
    map_landmarks_[next_landmark_id_] = {point, descriptor.clone()};
    active_kps_.push_back({next_landmark_id_, keypoint});
    next_landmark_id_++;
  }

  // Returns a pair of points3d and keypoints
  void get_point_keypoint_pairs(
    std::vector<cv::Point3f> & points, std::vector<cv::Point2f> & keypoints) const
  {
    points.clear();
    keypoints.clear();

    points.reserve(active_kps_.size());
    keypoints.reserve(active_kps_.size());

    for (auto & kp : active_kps_) {
      points.push_back(map_landmarks_.at(kp.landmark_id).pose);
      keypoints.push_back(kp.pt2d);
    }
  }

  void update_active_kps(const std::vector<TrackedKP> & kps) { active_kps_ = kps; }

  const std::vector<TrackedKP> & get_active_kps() const { return active_kps_; }

  void update_pose(const cv::Affine3d & pose) { current_pose_ = pose; }

  cv::Affine3d get_pose() const { return current_pose_; }

  void set_last_frame(const cv::Mat & frame) { last_frame_ = frame; }

private:
  std::map<long, Landmark> map_landmarks_;
  std::vector<TrackedKP> active_kps_;
  cv::Mat last_frame_;
  cv::Affine3d current_pose_;
  long next_landmark_id_ = 0;
};
}  // namespace mono_vo