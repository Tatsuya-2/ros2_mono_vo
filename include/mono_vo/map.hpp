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
    landmarks_.emplace(landmark.id, landmark);
  }

  const Landmark & get_landmark(long id) { return landmarks_.at(id); }

  /**
 * Extracts 2D-3D point correspondences from observations.
 *
 * This function iterates over a collection of observations, extracting the 2D image
 * points and corresponding 3D world points from the map's landmarks. The resulting
 * pairs of 2D-3D points are often used for Perspective-n-Point (PnP) estimation to
 * determine the camera pose.
 *
 * @param observations A vector of observations containing 2D keypoints and associated landmark IDs.
 * @return A pair of vectors: the first containing 2D points, and the second containing corresponding 3D points.
 */
  std::pair<std::vector<cv::Point2f>, std::vector<cv::Point3f>>
  get_observation_landmark_point_correspondences(const std::vector<Observation> & observations)
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

  void add_keyframe(const std::shared_ptr<KeyFrame> & keyframe)
  {
    RCLCPP_INFO(logger_, "Adding keyframe %ld", keyframe->id);
    keyframes_.emplace(keyframe->id, keyframe);
    last_keyframe_id_ = keyframe->id;
  }

  KeyFrame::Ptr & get_keyframe(long id) { return keyframes_.at(id); }

  KeyFrame::Ptr & get_last_keyframe() { return keyframes_.at(last_keyframe_id_); }

  const std::map<long, Landmark> & get_all_landmarks() const { return landmarks_; }

  const std::map<long, KeyFrame::Ptr> & get_all_keyframes() const { return keyframes_; }

  size_t num_landmarks() const { return landmarks_.size(); }

  size_t num_keyframes() const { return keyframes_.size(); }

  std::vector<cv::Point3f> get_landmark_points() const
  {
    std::vector<cv::Point3f> points;
    points.reserve(landmarks_.size());
    for (const auto & landmark : landmarks_) {
      points.push_back(landmark.second.pose_w);
    }
    return points;
  }

private:
  std::map<long, Landmark> landmarks_;
  std::map<long, KeyFrame::Ptr> keyframes_;
  rclcpp::Logger logger_;
  long last_keyframe_id_ = 0;
};
}  // namespace mono_vo