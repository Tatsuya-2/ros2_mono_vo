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
  KeyFrame(const cv::Affine3d & pose) : id(next_id_++), pose_wc(pose) {}

  bool isAffine3dDefault(const cv::Affine3d & pose, double eps = 1e-9)
  {
    return cv::norm(pose.matrix - cv::Affine3d().matrix) < eps;
  }
  // A constructor to create a KeyFrame from a temporary Frame object
  KeyFrame(const Frame & frame)
  {
    // assert that the frame is not empty
    assert(!frame.keypoints.empty());
    assert(!frame.descriptors.empty());
    assert(!frame.landmark_ids.empty());
    assert(!isAffine3dDefault(frame.pose_wc, 1e-9));
    id = next_id_++;
    pose_wc = frame.pose_wc;
    keypoints = frame.keypoints;
    descriptors = frame.descriptors;
    landmark_ids = frame.landmark_ids;
  }

  void add_observation(const cv::KeyPoint & keypoint, const cv::Mat & descriptor, long landmark_id)
  {
    keypoints.push_back(keypoint);
    descriptors.push_back(descriptor);
    landmark_ids.push_back(landmark_id);
  }

  std::vector<cv::Point2f> get_points_2d() const
  {
    std::vector<cv::Point2f> points_2d;
    points_2d.resize(keypoints.size());
    for (size_t i = 0; i < keypoints.size(); i++) {
      points_2d[i] = keypoints[i].pt;
    }
    return points_2d;
  }

  long id;
  cv::Affine3d pose_wc;  // Pose of the camera in the world (T_wc)
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  std::vector<long> landmark_ids;

private:
  static long next_id_;
};

long KeyFrame::next_id_ = 0;

}  // namespace mono_vo