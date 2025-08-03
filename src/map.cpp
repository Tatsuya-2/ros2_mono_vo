#include "mono_vo/map.hpp"

namespace mono_vo
{
Map::Map(rclcpp::Logger logger) : logger_(logger) {}

void Map::add_landmark(const Landmark & landmark)
{
  RCLCPP_INFO(logger_, "Adding landmark %ld", landmark.id);
  landmarks_.emplace(landmark.id, landmark);
}

const Landmark & Map::get_landmark(long id) { return landmarks_.at(id); }

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point3f>>
Map::get_observation_landmark_point_correspondences(const std::vector<Observation> & observations)
{
  std::vector<cv::Point2f> points_2d;
  std::vector<cv::Point3f> points_3d;
  points_3d.reserve(observations.size());
  points_2d.reserve(observations.size());
  for (const auto & obs : observations) {
    if (obs.landmark_id == -1) {
      continue;
    }
    points_2d.push_back(obs.keypoint.pt);
    points_3d.push_back(landmarks_.at(obs.landmark_id).pose_w);
  }
  points_2d.shrink_to_fit();
  points_3d.shrink_to_fit();
  return {points_2d, points_3d};
}

void Map::add_keyframe(const std::shared_ptr<KeyFrame> & keyframe)
{
  RCLCPP_INFO(logger_, "Adding keyframe %ld", keyframe->id);
  keyframes_.emplace(keyframe->id, keyframe);
  last_keyframe_id_ = keyframe->id;
}

KeyFrame::Ptr & Map::get_keyframe(long id) { return keyframes_.at(id); }

KeyFrame::Ptr & Map::get_last_keyframe() { return keyframes_.at(last_keyframe_id_); }

const std::map<long, Landmark> & Map::get_all_landmarks() const { return landmarks_; }

const std::map<long, KeyFrame::Ptr> & Map::get_all_keyframes() const { return keyframes_; }

size_t Map::num_landmarks() const { return landmarks_.size(); }

size_t Map::num_keyframes() const { return keyframes_.size(); }

std::vector<cv::Point3f> Map::get_landmark_points() const
{
  std::vector<cv::Point3f> points;
  points.reserve(landmarks_.size());
  for (const auto & landmark : landmarks_) {
    points.push_back(landmark.second.pose_w);
  }
  return points;
}

}  // namespace mono_vo