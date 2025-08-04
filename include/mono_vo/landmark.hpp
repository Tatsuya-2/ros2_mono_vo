#pragma once

#include <opencv2/core.hpp>

namespace mono_vo
{
struct Landmark
{
public:
  explicit Landmark(const cv::Point3f & pose_w, const cv::Mat & descriptor);

  long id;
  cv::Point3f pose_w;  // pose in world frame
  cv::Mat descriptor;

private:
  static long next_id_;
};

}  // namespace mono_vo