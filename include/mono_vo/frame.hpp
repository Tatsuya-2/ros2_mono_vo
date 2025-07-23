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
    points_2d.reserve(keypoints.size());
    for (const auto & keypoint : keypoints) {
      points_2d.push_back(keypoint.pt);
    }
    return points_2d;
  }

  void filter_observations_by_mask(const std::vector<uchar> & inlier_mask)
  {
    if (keypoints.size() != inlier_mask.size() || descriptors.rows != inlier_mask.size()) {
      throw std::runtime_error("Inlier mask, keypoints and descriptors must have the same size");
    }
    std::vector<cv::KeyPoint> in_kps;
    std::vector<cv::Mat> in_desc_rows;
    in_kps.reserve(inlier_mask.size());
    in_desc_rows.reserve(inlier_mask.size());
    for (size_t i = 0; i < inlier_mask.size(); ++i) {
      if (inlier_mask[i]) {
        in_kps.push_back(keypoints[i]);
        in_desc_rows.push_back(descriptors.row(i));
      }
    }
    keypoints = in_kps;
    cv::Mat in_descs;
    cv::vconcat(in_desc_rows, in_descs);
    descriptors = in_descs;
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