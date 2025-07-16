#pragma once

#include <opencv2/core.hpp>

namespace mono_vo
{
struct Landmark
{
  long id;
  cv::Point3f pose_w;  // pose in world frame
  cv::Mat descriptor;
};
}  // namespace mono_vo