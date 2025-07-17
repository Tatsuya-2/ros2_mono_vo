// mono_vo/frame.h
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

#include "mono_vo/landmark.hpp"

namespace mono_vo
{
// Temporary frame to store data before it is added to the map
class Frame
{
public:
  Frame(const cv::Mat & image) : id(next_id_++), image(image.clone()) {}

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
  cv::Mat image;
  cv::Affine3d pose_wc;  // Pose of the camera in the world (T_wc)
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  std::vector<long> landmark_ids;

private:
  static long next_id_;
};

long Frame::next_id_ = 0;
}  // namespace mono_vo