#pragma once

#include <opencv2/core.hpp>

namespace mono_vo
{
class Landmark
{
public:
  explicit Landmark(const cv::Point3f & pose_w, const cv::Mat & descriptor)
  : id(next_id_++), pose_w(pose_w), descriptor(descriptor)
  {
  }
  long id;
  cv::Point3f pose_w;  // pose in world frame
  cv::Mat descriptor;

private:
  static long next_id_;
};

long Landmark::next_id_ = 0;
}  // namespace mono_vo