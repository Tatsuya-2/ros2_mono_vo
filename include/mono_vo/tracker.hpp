#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
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
    rclcpp::Logger logger = rclcpp::get_logger("Tracker"))
  : map_(map), prev_frame_(cv::Mat()), feature_extractor_(feature_extractor), logger_(logger)
  {
  }

  Frame track_frame_with_optical_flow(const cv::Mat & new_image)
  {
    Frame new_frame{new_image};
    std::vector<cv::Point2f> prev_pts_2d = prev_frame_.get_points_2d();
    std::vector<cv::Point2f> new_pts_2d;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(
      prev_frame_.image, new_frame.image, prev_pts_2d, new_pts_2d, status, err);
    for (size_t i = 0; i < prev_pts_2d.size(); ++i) {
      if (status[i] && err[i] < tracking_error_thresh_) {
        new_frame.add_observation(
          cv::KeyPoint(new_pts_2d[i], 1), prev_frame_.observations[i].descriptor,
          prev_frame_.observations[i].landmark_id);
      }
    }
    new_frame.is_tracked = true;
    return new_frame;
  }

  // check if pose difference between frames is significant
  bool significant_motion(const Frame & frame)
  {
    // get relative pose
    cv::Affine3d relative_pose = prev_frame_.pose_wc.inv() * frame.pose_wc;

    if (cv::norm(relative_pose.matrix) > max_translation_from_keyframe_) {
      RCLCPP_WARN(logger_, "translation exceeds threshold");
      return true;
    }

    cv::Matx33d rotation_matrix = relative_pose.rotation();

    // Convert the rotation matrix to an axis-angle rotation vector
    cv::Vec3d rotation_vector;
    cv::Rodrigues(rotation_matrix, rotation_vector);

    // The magnitude of the rotation vector is the angle in radians
    if (cv::norm(rotation_vector) > max_rotation_from_keyframe_) {
      RCLCPP_WARN(logger_, "rotation exceeds threshold");
      return true;
    }

    return false;
  }

  bool should_add_keyframe(const Frame & frame)
  {
    if (frame.observations.size() < min_observations_before_triangulation_) {
      RCLCPP_WARN(logger_, "Not enough 3D points for triangulation");
      return true;
    }

    if (tracking_count_from_keyframe_ > max_tracking_after_keyframe_) {
      RCLCPP_WARN(logger_, "Not enough tracking after keyframe");
      return true;
    }

    if (significant_motion(frame)) {
      RCLCPP_WARN(logger_, "Significant motion");
      return true;
    }

    return false;

    // return frame.observations.size() < min_observations_before_triangulation_ ||
    //        tracking_count_from_keyframe_++ > max_tracking_after_keyframe_ ||
    //        significant_motion(frame);
  }

  TrackerState get_state() const { return state_; }

  std::optional<cv::Affine3d> update(const Frame & frame, const cv::Mat & K, const cv::Mat & d)
  {
    // set first frame
    if (state_ == TrackerState::INITIALIZING) {
      prev_frame_ = frame;
      state_ = TrackerState::TRACKING;
      return std::nullopt;
    }

    // --- track points with optical flow ---
    Frame new_frame = track_frame_with_optical_flow(frame.image);

    cv::Mat img_matches = utils::draw_matched_frames(prev_frame_, new_frame);
    cv::imshow("Matches", img_matches);
    cv::waitKey(1);

    RCLCPP_INFO(logger_, "Tracked %zu points using optical flow", new_frame.observations.size());

    // check if enough points were tracked
    if (new_frame.observations.size() < min_tracked_points_) {
      RCLCPP_WARN(logger_, "Not enough keypoints were tracked");
      // TODO: handle logic here
      return std::nullopt;
    }

    // --- solve PnP ---
    // get new frame tracked 2D and corresponding 3D points
    auto [new_2dps, new_3dps] =
      map_->get_observation_landmark_point_correspondences(new_frame.observations);

    RCLCPP_INFO(logger_, "Got %zu 3D point correspondences from tracking", new_3dps.size());

    // Transform that brings point in world to camera frame
    cv::Mat rvec;     // rotation vector in 3x1 format (Rodrigues format)
    cv::Mat tvec;     // translation vector in 3x1 format
    cv::Mat inliers;  // indices of inliers
    cv::solvePnPRansac(new_3dps, new_2dps, K, d, rvec, tvec, false, 100, 8.0, 0.99, inliers);

    RCLCPP_INFO(logger_, "Solved PnP with %d inliers", inliers.rows);

    // filter new pose inliers
    new_frame.filter_observations_by_mask(inliers);

    // get camera pose in world frame
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    cv::Mat R_cw = R.t();
    cv::Mat t_cw = -R_cw * tvec;
    new_frame.pose_wc = cv::Affine3d(R_cw, t_cw);

    tracking_count_from_keyframe_++;
    if (should_add_keyframe(new_frame)) {
      // check if has enough baseline
    }

    std::stringstream ss;
    ss << "Camera pose rotation: \n" << R_cw << std::endl;
    ss << "Camera pose translation: \n" << t_cw << std::endl;
    RCLCPP_INFO(logger_, "%s", ss.str().c_str());

    // update last frame
    prev_frame_ = std::move(new_frame);
    return prev_frame_.pose_wc;
  }

private:
  std::shared_ptr<Map> map_;
  TrackerState state_ = TrackerState::INITIALIZING;
  Frame prev_frame_;
  FeatureExtractor::Ptr feature_extractor_;
  rclcpp::Logger logger_;
  float tracking_error_thresh_ = 30.0;
  size_t min_observations_before_triangulation_ = 300;
  size_t min_tracked_points_ = 10;
  size_t max_tracking_after_keyframe_ = 10;
  size_t tracking_count_from_keyframe_ = 0;
  double max_rotation_from_keyframe_ = M_PI / 12.0;  // in radians (15 degrees)
  double max_translation_from_keyframe_ = 0.3;       // in meters
};
}  // namespace mono_vo