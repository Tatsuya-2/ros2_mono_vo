#pragma once

#include <memory>
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
  using Ptr = std::shared_ptr<Map>;

  Map(rclcpp::Logger logger = rclcpp::get_logger("Map"));

  void add_landmark(const Landmark & landmark);

  const Landmark & get_landmark(long id);

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
  get_observation_to_landmark_point_correspondences(const std::vector<Observation> & observations);

  void add_keyframe(const std::shared_ptr<KeyFrame> & keyframe);

  KeyFrame::Ptr & get_keyframe(long id);

  KeyFrame::Ptr & get_last_keyframe();

  const std::map<long, Landmark> & get_all_landmarks() const;

  const std::map<long, KeyFrame::Ptr> & get_all_keyframes() const;

  size_t num_landmarks() const;

  size_t num_keyframes() const;

  std::vector<cv::Point3f> get_landmark_points() const;

private:
  std::map<long, Landmark> landmarks_;
  std::map<long, KeyFrame::Ptr> keyframes_;
  rclcpp::Logger logger_;
  long last_keyframe_id_ = 0;
};
}  // namespace mono_vo