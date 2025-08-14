#pragma once

#include <cmath>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <rclcpp/logging.hpp>
#include <vector>

#include "mono_vo/feature_processor.hpp"
#include "mono_vo/frame.hpp"
#include "mono_vo/keyframe.hpp"
#include "mono_vo/landmark.hpp"
#include "mono_vo/map.hpp"
#include "mono_vo/ros_parameter_handler.hpp"
#include "mono_vo/utils.hpp"

namespace mono_vo
{
enum class TrackerState
{
  INITIALIZING,
  TRACKING,
  LOST
};

class Tracker
{
public:
  Tracker(
    Map::Ptr map,
    FeatureProcessor::Ptr feature_extractor = std::make_shared<FeatureProcessor>(1000),
    rclcpp::Logger logger = rclcpp::get_logger("Tracker"));

  void configure_parameters(RosParameterHandler & param_handler);

  /**
   * Tracks the given new image using optical flow.
   *
   * @param new_image The new image to track.
   * @return The tracked frame with the new image and the tracked observations.
   */
  Frame track_frame_with_optical_flow(const cv::Mat & new_image);

  /**
   * Checks if the current frame has significant motion compared to the last keyframe.
   *
   * This function computes the relative pose between the last keyframe and the provided frame.
   * It evaluates both the translation and rotation components of the relative pose. If either the
   * translation exceeds `max_translation_from_keyframe_` or the rotation exceeds 
   * `max_rotation_from_keyframe_`, the function returns true, indicating
   * significant motion. Otherwise, it returns false.
   *
   * @param frame The current frame whose motion is being evaluated.
   * @return True if significant motion is detected, otherwise false.
   */
  bool has_significant_motion(const Frame & frame);

  /**
   * Determines if a new keyframe should be added based on the current frame.
   *
   * This function evaluates various conditions to decide if a new keyframe
   * should be added to the map. It checks if the current frame has sufficient
   * observations for triangulation, if the tracking count since the last keyframe
   * exceeds a maximum threshold, and if there has been significant motion since
   * the last keyframe. If any of these conditions are met, the function returns
   * true, indicating that a new keyframe should be added.
   *
   * @param frame The current frame being evaluated.
   * @return True if a new keyframe should be added, otherwise false.
   */
  bool should_add_keyframe(const Frame & frame);

  std::vector<cv::Point3f> triangulate_points(
    const cv::Affine3d & pose_ref_cw, const cv::Affine3d & pose_cur_cw, const cv::Mat & K,
    const std::vector<cv::Point2f> & pts_ref, const std::vector<cv::Point2f> & pts_cur,
    std::vector<uchar> & inliers);

  /**
   * Adds a new keyframe to the map using the provided frame and camera intrinsic matrix.
   *
   * This function first clears and re-extracts observations from the given frame. It then finds
   * feature matches between the new frame and the previous keyframe. Using these matches, it
   * computes the 3D points through triangulation and adds them to the map as landmarks. If a
   * landmark already exists in the previous keyframe for a matched point, it reuses the existing
   * landmark ID. Finally, the new frame is added as a keyframe to the map, and the tracking count
   * from the keyframe is reset.
   *
   * @param frame The current frame to be added as a keyframe.
   * @param K The camera intrinsic matrix.
   */
  void add_new_keyframe(Frame & frame, const cv::Mat & K);

  /**
   * @brief Determines if there is sufficient parallax between the last keyframe and the current frame.
   *
   * This function computes the homography and fundamental matrix between the 2D points observed in 
   * the last keyframe and the current frame. It evaluates the quality of the fundamental matrix 
   * model compared to the homography model to ensure that there is enough parallax for reliable 
   * triangulation.
   *
   * @param frame The current frame for which parallax needs to be checked.
   * @return True if there is sufficient parallax, false otherwise.
   */
  bool has_parallax(const Frame & frame);

  TrackerState get_state() const;

  void reset();

  /**
   * Updates the tracker with a new frame.
   *
   * @param frame the new frame
   * @param K the intrinsic matrix
   * @param d the distortion coefficients
   *
   * @returns the new camera pose in world coordinates if the frame was successfully tracked
   *          or an empty optional if the frame couldn't be tracked
   *
   * The following steps are performed:
   * - track points with optical flow
   * - check if enough points were tracked
   * - solve PnP to get the camera pose in world frame
   * - check if the camera pose has enough parallax to be considered a keyframe
   * - add the new frame as a keyframe if it has enough parallax
   * - update the last frame
   */
  std::optional<cv::Affine3d> update(const Frame & frame, const cv::Mat & K, const cv::Mat & d);

private:
  Map::Ptr map_;
  TrackerState state_ = TrackerState::INITIALIZING;
  Frame prev_frame_;
  FeatureProcessor::Ptr feature_processor_;
  rclcpp::Logger logger_;
  float tracking_error_thresh_ = 30.0;                       // px LK tracking error threshold
  int64_t min_observations_before_triangulation_ = 100;      // trigger for keyframe addition
  int64_t min_tracked_points_ = 10;                          // below which tracker is declared lost
  int64_t max_tracking_after_keyframe_ = 10;                 // trigger for keyframe addition
  int64_t tracking_count_from_keyframe_ = 0;                 // tracking count since last keyframe
  double max_rotation_from_keyframe_ = M_PI * 15.0 / 180.0;  // radians (15 degrees)
  double max_translation_from_keyframe_ = 1.0;               // in meters
  double ransac_reproj_thresh_ = 1.0;  // px reprojection threshold for H/F model
  double model_score_thresh_ = 0.85;   // h/f score
  double f_inlier_thresh_ = 0.5;       // funandamental inlier threshold
  double lowes_distance_ratio_ = 0.7;  // Lowe's distance ratio for finding good matches
};
}  // namespace mono_vo