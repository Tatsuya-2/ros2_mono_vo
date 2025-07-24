// mono_vo/frame.h
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

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
class Frame
{
public:
  explicit Frame(const cv::Mat & image) : id(next_id_++), image(image.clone()) {}

  void add_observation(const cv::KeyPoint & keypoint, const cv::Mat & descriptor, long landmark_id)
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
  }

  long id;
  cv::Mat image;
  cv::Affine3d pose_wc;  // Pose of the camera in the world (T_wc)
  // std::vector<cv::KeyPoint> keypoints;
  // cv::Mat descriptors;
  // std::vector<long> landmark_ids;
  std::vector<Observation> observations;

private:
  static long next_id_;
};

long Frame::next_id_ = 0;
}  // namespace mono_vo