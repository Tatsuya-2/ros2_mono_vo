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
    return new_frame;
  }

  TrackerState get_state() const { return state_; }

  void update(const Frame & frame, const cv::Mat & K, const cv::Mat & d)
  {
    // set first frame
    if (state_ == TrackerState::INITIALIZING) {
      prev_frame_ = frame;
      state_ = TrackerState::TRACKING;
      return;
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
      return;
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

    std::stringstream ss;
    ss << "Camera pose rotation: \n" << R_cw << std::endl;
    ss << "Camera pose translation: \n" << t_cw << std::endl;
    RCLCPP_INFO(logger_, "%s", ss.str().c_str());

    // update last frame
    prev_frame_ = new_frame;
  }

private:
  std::shared_ptr<Map> map_;
  TrackerState state_ = TrackerState::INITIALIZING;
  Frame prev_frame_;
  FeatureExtractor::Ptr feature_extractor_;
  rclcpp::Logger logger_;
  float tracking_error_thresh_ = 30.0;
  size_t min_tracked_points_ = 10;
};
}  // namespace mono_vo