#pragma once

#include <opencv2/opencv.hpp>
#include <rclcpp/logging.hpp>

#include "mono_vo/frame.hpp"
#include "mono_vo/keyframe.hpp"
#include "mono_vo/landmark.hpp"

namespace mono_vo
{
class Map
{
public:
  Map(rclcpp::Logger logger = rclcpp::get_logger("Map")) : logger_(logger) {};

  void add_landmark(const Landmark & landmark)
  {
    RCLCPP_INFO(logger_, "Adding landmark %ld", landmark.id);
    landmarks_[landmark.id] = landmark;
  }

  const Landmark & get_landmark(long id) { return landmarks_.at(id); }

  void add_keyframe(const std::shared_ptr<KeyFrame> & keyframe)
  {
    RCLCPP_INFO(logger_, "Adding keyframe %ld", keyframe->id);
    keyframes_[keyframe->id] = keyframe;
  }

  const KeyFrame::Ptr & get_keyframe(long id) { return keyframes_.at(id); }

  const std::map<long, Landmark> & get_all_landmarks() const { return landmarks_; }

  const std::map<long, KeyFrame::Ptr> & get_all_keyframes() const { return keyframes_; }

  size_t num_landmarks() const { return landmarks_.size(); }

  size_t num_keyframes() const { return keyframes_.size(); }

private:
  std::map<long, Landmark> landmarks_;
  std::map<long, KeyFrame::Ptr> keyframes_;
  rclcpp::Logger logger_;
};
}  // namespace mono_vo