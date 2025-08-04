#include "mono_vo/keyframe.hpp"

namespace mono_vo
{

long KeyFrame::next_id_ = 0;

KeyFrame::KeyFrame(const cv::Affine3d & pose) : id(next_id_++), pose_wc(pose) {}

bool KeyFrame::isAffine3dDefault(const cv::Affine3d & pose, double eps)
{
  return cv::norm(pose.matrix - cv::Affine3d().matrix) < eps;
}

KeyFrame::KeyFrame(const Frame & frame)
{
  // assert that the frame is not empty
  assert(!frame.observations.empty());
  assert(!isAffine3dDefault(frame.pose_wc, 1e-9));
  id = next_id_++;
  pose_wc = frame.pose_wc;
  observations = frame.observations;
  landmark_id_to_index_.reserve(observations.size());
  for (size_t i = 0; i < observations.size(); ++i) {
    if (observations[i].landmark_id != -1) {
      landmark_id_to_index_[observations[i].landmark_id] = i;
    }
  }
}

void KeyFrame::add_observation(
  const cv::KeyPoint & keypoint, const cv::Mat & descriptor, long landmark_id)
{
  observations.emplace_back(keypoint, descriptor, landmark_id);
  if (landmark_id != -1) {
    landmark_id_to_index_[landmark_id] = observations.size() - 1;
  }
}

std::vector<cv::Point2f> KeyFrame::get_points_2d(ObservationFilter filter_type) const
{
  std::vector<cv::Point2f> points_2d;
  points_2d.reserve(observations.size());
  for (const auto & obs : observations) {
    switch (filter_type) {
      case ObservationFilter::WITHOUT_LANDMARKS:
        if (obs.landmark_id == -1) points_2d.push_back(obs.keypoint.pt);
        break;
      case ObservationFilter::WITH_LANDMARKS:
        if (obs.landmark_id != -1) points_2d.push_back(obs.keypoint.pt);
        break;
      case ObservationFilter::ALL:
        points_2d.push_back(obs.keypoint.pt);
        break;

      default:
        throw std::runtime_error("Invalid filter type in KeyFrame::get_points_2d()");
        break;
    }
  }
  points_2d.shrink_to_fit();
  return points_2d;
}

std::vector<cv::Point2f> KeyFrame::get_points_2d_for_landmarks(
  const std::vector<long> & landmark_ids) const
{
  std::vector<cv::Point2f> points_2d;
  points_2d.reserve(landmark_ids.size());
  for (const auto landmark_id : landmark_ids) {
    if (auto it = landmark_id_to_index_.find(landmark_id); it != landmark_id_to_index_.end()) {
      const size_t index = it->second;
      points_2d.push_back(observations[index].keypoint.pt);
    }
  }
  points_2d.shrink_to_fit();
  return points_2d;
}

std::optional<Observation> KeyFrame::get_observation_for_landmark(long landmark_id) const
{
  auto it = landmark_id_to_index_.find(landmark_id);
  if (it != landmark_id_to_index_.end()) {
    return observations[it->second];
  }
  return std::nullopt;
}

std::vector<cv::KeyPoint> KeyFrame::get_keypoints() const
{
  std::vector<cv::KeyPoint> keypoints;
  keypoints.reserve(observations.size());
  for (const auto & obs : observations) {
    keypoints.push_back(obs.keypoint);
  }
  return keypoints;
}

cv::Mat KeyFrame::get_descriptors() const
{
  cv::Mat descriptors;
  if (observations.empty()) {
    return descriptors;
  }

  const auto & d = observations[0].descriptor;
  descriptors.create(observations.size(), d.cols, d.type());

  for (size_t i = 0; i < observations.size(); ++i) {
    observations[i].descriptor.copyTo(descriptors.row(i));
  }
  return descriptors;
}

std::vector<Observation> KeyFrame::get_observations(ObservationFilter filter_type) const
{
  if (filter_type == ObservationFilter::ALL) {
    return observations;
  }
  std::vector<Observation> valid_obs;
  valid_obs.reserve(observations.size());
  for (const auto & obs : observations) {
    switch (filter_type) {
      case ObservationFilter::WITH_LANDMARKS:
        if (obs.landmark_id != -1) valid_obs.push_back(obs);
        break;

      case ObservationFilter::WITHOUT_LANDMARKS:
        if (obs.landmark_id == -1) valid_obs.push_back(obs);
        break;

      default:
        throw std::runtime_error("Invalid filter type for Frame::get_observations()");
        break;
    }
  }
  return valid_obs;
}

void KeyFrame::clear_observations()
{
  observations.clear();
  landmark_id_to_index_.clear();
}

}  // namespace mono_vo