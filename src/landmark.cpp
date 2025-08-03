#include "mono_vo/landmark.hpp"

namespace mono_vo
{
long Landmark::next_id_ = 0;

Landmark::Landmark(const cv::Point3f & pose_w, const cv::Mat & descriptor)
: id(next_id_++), pose_w(pose_w), descriptor(descriptor)
{
}
}  // namespace mono_vo