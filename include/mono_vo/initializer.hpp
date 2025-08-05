#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <rclcpp/logging.hpp>
#include <vector>

#include "mono_vo/feature_processor.hpp"
#include "mono_vo/frame.hpp"
#include "mono_vo/map.hpp"
#include "mono_vo/match_data.hpp"
#include "mono_vo/utils.hpp"

namespace mono_vo
{

class Initializer
{
public:
  enum class State
  {
    OBTAINING_REF,
    INITIALIZING,
    INITIALIZED
  };

  Initializer(
    std::shared_ptr<Map> map,
    FeatureProcessor::Ptr feature_extractor = std::make_shared<FeatureProcessor>(1000),
    rclcpp::Logger logger = rclcpp::get_logger("Initializer"));

  bool is_initalized();

  void reset();

  bool good_keypoint_distribution(const Frame & frame);

  bool check_parallax(const std::vector<cv::Point2f> & pts1, const std::vector<cv::Point2f> & pts2);

  /**
   * Triangulate points given reference and current frames.
   *
   * The function takes two sets of 2D points, ref_points and cur_points, and computes the corresponding 3D points.
   * The points are triangulated using the projection matrices P_ref and P_cur, and the inliers are
   * checked using a chirality check.
   *
   * The output is a 3d point vector, whose size() is the number of valid 3D points.
   *
   * @param K The camera intrinsic matrix.
   * @param R The rotation matrix from the reference frame to the current frame.
   * @param t The translation vector from the reference frame to the current frame.
   * @param ref_points The 2D points in the reference frame.
   * @param cur_points The 2D points in the current frame.
   * @param inliers The inliers of the triangulation.
   *
   * @return A 3d point vector.
   */
  std::vector<cv::Point3f> traingulate_points(
    const cv::Mat & K, const cv::Mat & R_cw, const cv::Mat & t_cw,
    const std::vector<cv::Point2f> & ref_points, const std::vector<cv::Point2f> & cur_points,
    std::vector<uchar> & inliers);

  std::optional<Frame> try_initializing(const Frame & frame, const cv::Mat & K);

private:
  std::shared_ptr<Map> map_;
  rclcpp::Logger logger_;
  State state_;
  Frame ref_frame_;
  float distribution_thresh_;
  double lowes_distance_ratio_ = 0.7;
  double min_matches_for_parallax_ = 100;
  double ransac_thresh_h_ = 2.0;      // px homography RANSAC threshold
  double ransac_thresh_f_ = 1.0;      // px fundamental RANSAC threshold
  double f_inlier_thresh_ = 0.5;      // fundamental inlier threshold ratio
  double model_score_thresh_ = 0.56;  // max H/F ratio

  double current_min_model_score_ = 100.0;  // min H/F ratio for debug

  FeatureProcessor::Ptr feature_processor_;
};
}  // namespace mono_vo