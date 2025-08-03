#pragma once

#include <cmath>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <rclcpp/logging.hpp>
#include <vector>

#include "mono_vo/feature_extractor.hpp"
#include "mono_vo/frame.hpp"
#include "mono_vo/keyframe.hpp"
#include "mono_vo/landmark.hpp"
#include "mono_vo/map.hpp"
#include "mono_vo/utils.hpp"

namespace mono_vo
{
enum class TrackerState
{
  INITIALIZING,
  TRACKING
};

class Tracker
{
public:
  Tracker(
    std::shared_ptr<Map> map,
    FeatureExtractor::Ptr feature_extractor = std::make_shared<FeatureExtractor>(1000),
    rclcpp::Logger logger = rclcpp::get_logger("Tracker"));

  Frame track_frame_with_optical_flow(const cv::Mat & new_image);

  bool significant_motion(const Frame & frame);

  bool should_add_keyframe(const Frame & frame);

  std::vector<cv::Point3f> triangulate_points(
    const cv::Affine3d & pose_ref_cw, const cv::Affine3d & pose_cur_cw, const cv::Mat & K,
    const std::vector<cv::Point2f> & pts_ref, const std::vector<cv::Point2f> & pts_cur,
    std::vector<uchar> & inliers);

  void add_new_keyframe(Frame & frame, const cv::Mat & K);

  bool has_parallax(const Frame & frame);

  TrackerState get_state() const;

  std::optional<cv::Affine3d> update(const Frame & frame, const cv::Mat & K, const cv::Mat & d);

private:
  std::shared_ptr<Map> map_;
  TrackerState state_ = TrackerState::INITIALIZING;
  Frame prev_frame_;
  FeatureExtractor::Ptr feature_extractor_;
  rclcpp::Logger logger_;
  float tracking_error_thresh_ = 30.0;
  size_t min_observations_before_triangulation_ = 100;
  size_t min_tracked_points_ = 10;
  size_t max_tracking_after_keyframe_ = 10;
  size_t tracking_count_from_keyframe_ = 0;
  double max_rotation_from_keyframe_ = M_PI / 12.0;  // in radians (15 degrees)
  double max_translation_from_keyframe_ = 1.0;       // in meters
  double ransac_reprojection_thresh_ = 3.0;          // in pixels
  double model_score_thresh_ = 0.85;                 // h/f score
  double f_inlier_thresh_ = 0.5;
  double lowes_distance_ratio_ = 0.7;
};
}  // namespace mono_vo