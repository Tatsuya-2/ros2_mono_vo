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
  landmark_id_to_observation.reserve(observations.size());
  for (auto & obs : observations) {
    if (obs.landmark_id != -1) {
      landmark_id_to_observation[obs.landmark_id] = &obs;
    }
  }
}

void KeyFrame::add_observation(
  const cv::KeyPoint & keypoint, const cv::Mat & descriptor, long landmark_id)
{
  observations.emplace_back(keypoint, descriptor, landmark_id);
  if (landmark_id != -1) {
    landmark_id_to_observation[landmark_id] = &observations.back();
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
    if (auto obs = landmark_id_to_observation.at(landmark_id); obs != nullptr) {
      points_2d.push_back(obs->keypoint.pt);
    }
  }
  points_2d.shrink_to_fit();
  return points_2d;
}

const Observation * KeyFrame::get_observation_for_landmark(long landmark_id) const
{
  return landmark_id_to_observation.at(landmark_id);
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

void KeyFrame::clear_observations() { observations.clear(); }

}  // namespace mono_vo