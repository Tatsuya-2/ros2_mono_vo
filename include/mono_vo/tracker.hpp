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
    auto prev_pts_2d = prev_frame_.get_points_2d(true);
    auto prev_observations = prev_frame_.get_observations(true);
    std::vector<cv::Point2f> new_pts_2d;
    std::vector<uchar> status;
    std::vector<float> err;
    std::vector<cv::Point2f> prev_pts_2d_filtered;

    cv::calcOpticalFlowPyrLK(
      prev_frame_.image, new_frame.image, prev_pts_2d, new_pts_2d, status, err);
    for (size_t i = 0; i < prev_pts_2d.size(); ++i) {
      if (status[i] && err[i] < tracking_error_thresh_) {
        prev_pts_2d_filtered.push_back(prev_pts_2d[i]);  // TODO: remove, for debug only
        new_frame.add_observation(
          cv::KeyPoint(new_pts_2d[i], 1), prev_observations[i].descriptor,
          prev_observations[i].landmark_id);
        RCLCPP_INFO(logger_, "landmark id: %ld", new_frame.observations.back().landmark_id);
      }
    }

    RCLCPP_INFO(logger_, "tracked %d points", new_frame.observations.size());

    // TODO: remove, for debug only
    cv::Mat img_matches = utils::draw_matched_points(
      prev_frame_.image, new_frame.image, prev_pts_2d_filtered, new_frame.get_points_2d(true));
    cv::imshow("Matches", img_matches);
    cv::waitKey(0);

    new_frame.is_tracked = true;
    return new_frame;
  }

  // check if pose difference between frames is significant
  bool significant_motion(const Frame & frame)
  {
    // get relative pose
    KeyFrame::Ptr prev_kframe = map_->get_last_keyframe();
    cv::Affine3d relative_pose = prev_kframe->pose_wc.inv() * frame.pose_wc;

    double translation = cv::norm(relative_pose.translation());

    if (translation > max_translation_from_keyframe_) {
      RCLCPP_WARN(logger_, "translation exceeds threshold: %lf", translation);
      return true;
    }
    // Convert the rotation matrix to an axis-angle rotation vector
    cv::Matx33d R = relative_pose.rotation();
    double trace = R(0, 0) + R(1, 1) + R(2, 2);
    double rotation = std::acos((trace - 1.0) / 2.0);  // angle in radians

    // The magnitude of the rotation vector is the angle in radians
    if (rotation > max_rotation_from_keyframe_) {
      RCLCPP_WARN(logger_, "rotation exceeds threshold: %lf", rotation);
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

  void traingulate_new_points(Frame & frame)
  {
    // detect new features from frame

    // TODO store all keypoints in keyframes, regardless of matched.
    // find matches with the previous keyframe points that dont have landmarks.

    // use recovered pose from PnP and previous keyframe pose to get projection matrices

    // triangulate new 3D points and add to map

    // filter outlier 2D points from frame and set last_frame

    // set new keyframe with all detected keypoints

    // reset the tracking count
  }

  bool has_parallax(const Frame & frame)
  {
    auto pts1 = map_->get_last_keyframe()->get_points_2d_for_landmarks(frame.get_landmark_ids());
    auto pts2 = frame.get_points_2d();
    // calculate homography
    std::vector<uchar> inliers_h;
    cv::findHomography(pts1, pts2, cv::RANSAC, ransac_reprojection_thresh_, inliers_h);
    int score_h = cv::countNonZero(inliers_h);

    // calculate fundamental
    std::vector<uchar> inliers_f;
    cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, ransac_reprojection_thresh_, 0.99, inliers_f);
    int score_f = cv::countNonZero(inliers_f);

    RCLCPP_INFO(logger_, "score h: %d, score f: %d", score_h, score_f);
    // check 1: min inliers
    if (static_cast<double>(score_f) / pts1.size() < f_inlier_thresh_) {
      return false;
    }

    auto model_score = static_cast<double>(score_h) / static_cast<double>(score_f);
    RCLCPP_INFO(logger_, "score_h/score_f = %lf", model_score);

    // check 2: Is the Fundamental Matrix a significantly better model?
    // The ratio score_H / score_F should be low.
    // A high ratio means Homography explains the data almost as well as Fundamental Matrix.
    if (model_score > model_score_thresh_) {
      return false;
    }

    return true;
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
    // new_frame.filter_observations_by_mask(inliers);

    // get camera pose in world frame
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    new_frame.pose_wc = cv::Affine3d(R, tvec).inv();

    // tracking_count_from_keyframe_++;
    // if (should_add_keyframe(new_frame)) {
    //   // check if has enough parallax
    //   if (has_parallax(new_frame)) {
    //     RCLCPP_INFO(logger_, "Has enough parallax, adding keyframe");
    //   }
    // }

    std::stringstream ss;
    ss << "Camera pose transform wc: \n" << new_frame.pose_wc.matrix << std::endl;
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
  size_t min_observations_before_triangulation_ = 100;
  size_t min_tracked_points_ = 10;
  size_t max_tracking_after_keyframe_ = 10;
  size_t tracking_count_from_keyframe_ = 0;
  double max_rotation_from_keyframe_ = M_PI / 12.0;  // in radians (15 degrees)
  double max_translation_from_keyframe_ = 1.0;       // in meters
  double ransac_reprojection_thresh_ = 3.0;          // in pixels
  double model_score_thresh_ = 0.95;                 // h/f score
  double f_inlier_thresh_ = 0.5;
};
}  // namespace mono_vo