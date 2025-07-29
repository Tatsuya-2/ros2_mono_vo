#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <rclcpp/logging.hpp>
#include <vector>

#include "mono_vo/feature_extractor.hpp"
#include "mono_vo/frame.hpp"
#include "mono_vo/map.hpp"
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
    FeatureExtractor::Ptr feature_extractor = std::make_shared<FeatureExtractor>(1000),
    rclcpp::Logger logger = rclcpp::get_logger("Initializer"))
  : map_(map),
    logger_(logger),
    state_(State::OBTAINING_REF),
    ref_frame_(cv::Mat()),
    distribution_thresh_(0.5f),
    feature_extractor_(feature_extractor)
  {
  }

  bool is_initalized() { return state_ == State::INITIALIZED; }

  void reset() { state_ = State::OBTAINING_REF; }

  bool good_keypoint_distribution(const Frame & frame)
  {
    RCLCPP_INFO(logger_, "totals kps: %ld", frame.observations.size());

    // check distribution in a grid across the image
    cv::Mat grid = cv::Mat::zeros(frame.image.rows / 50, frame.image.cols / 50, CV_8U);
    int occupied_cells = 0;
    for (auto & obs : frame.observations) {
      int r = obs.keypoint.pt.y / 50, c = obs.keypoint.pt.x / 50;
      if (!grid.at<uchar>(r, c)) {
        grid.at<uchar>(r, c) = 1;
        occupied_cells++;
      }
    }
    auto total_cells = grid.cols * grid.rows;
    RCLCPP_INFO(logger_, "occupied cells: %d total cells: %d", occupied_cells, total_cells);
    auto occupancy = static_cast<float>(occupied_cells) / total_cells;
    RCLCPP_INFO(logger_, "occupancy: %f", occupancy);
    if (occupancy > distribution_thresh_) {
      return true;
    }
    return false;
  }

  bool check_parallax(const std::vector<cv::Point2f> & pts1, const std::vector<cv::Point2f> & pts2)
  {
    // calculate homography
    std::vector<uchar> inliers_h;
    cv::findHomography(pts1, pts2, cv::RANSAC, ransac_thresh_h_, inliers_h);
    int score_h = cv::countNonZero(inliers_h);

    // calculate fundamental
    std::vector<uchar> inliers_f;
    cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, ransac_thresh_f_, 0.99, inliers_f);
    int score_f = cv::countNonZero(inliers_f);

    RCLCPP_INFO(logger_, "score h: %d, score f: %d", score_h, score_f);
    // check 1: min inliers
    if (static_cast<double>(score_f) / pts1.size() < f_inlier_thresh_) {
      return false;
    }

    auto model_score = static_cast<double>(score_h) / static_cast<double>(score_f);
    RCLCPP_INFO(logger_, "score_h/score_f = %lf", model_score);

    current_min_model_score_ = std::min(current_min_model_score_, model_score);
    RCLCPP_INFO(logger_, "current_min_model_score = %lf", current_min_model_score_);

    // check 2: Is the Fundamental Matrix a significantly better model?
    // The ratio score_H / score_F should be low.
    // A high ratio means Homography explains the data almost as well as Fundamental Matrix.
    if (model_score > model_score_thresh_) {
      return false;
    }

    return true;
  }

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
    std::vector<uchar> & inliers)
  {
    // get projection matrices
    cv::Mat P_ref = K * cv::Mat::eye(3, 4, CV_64F);  // P_ref = K * [I | 0]

    cv::Mat Rt_cw;
    cv::hconcat(R_cw, t_cw, Rt_cw);
    cv::Mat P_cur = K * Rt_cw;  // P_cur = K * [R | t]

    cv::Mat pts4d_h;  // Output is 4xN matrix of homogeneous 3D points
    cv::triangulatePoints(P_ref, P_cur, ref_points, cur_points, pts4d_h);

    RCLCPP_INFO(logger_, "Triangulated %d 3D points", pts4d_h.cols);

    // convert to cartesian
    cv::Mat pts3d;
    cv::convertPointsFromHomogeneous(pts4d_h.t(), pts3d);

    // chirality check:check if point is in front of both reference and current camera frame
    auto is_infront = [&R_cw, &t_cw](const cv::Point3f & p3d) {
      // check if point is in front of the reference camera
      if (p3d.z <= 0) {
        return false;
      }

      // Transform the 3D point into the current camera's coordinate system.
      cv::Mat p3d_mat = (cv::Mat_<double>(3, 1) << p3d.x, p3d.y, p3d.z);
      cv::Mat p3d_cur_mat = R_cw * p3d_mat + t_cw;
      return p3d_cur_mat.at<double>(2, 0) > 0;
    };

    inliers.clear();
    inliers.reserve(pts3d.rows);
    std::vector<cv::Point3f> points_3d;
    for (int i = 0; i < pts3d.rows; ++i) {
      const cv::Point3f & p3d_ref = pts3d.at<cv::Point3f>(i);
      if (is_infront(p3d_ref)) {
        inliers.push_back(1);
        points_3d.push_back(p3d_ref);
      } else {
        inliers.push_back(0);
      }
    }

    RCLCPP_INFO(
      logger_, "Triangulation complete. Kept %zu / %d valid points.", points_3d.size(), pts3d.rows);

    return points_3d;
  }

  std::optional<Frame> try_initializing(const Frame & frame, const cv::Mat & K)
  {
    if (state_ == State::INITIALIZED) {
      return ref_frame_;
    }

    Frame cur_frame{frame};
    cur_frame.extract_features(feature_extractor_);

    if (state_ == State::OBTAINING_REF) {
      if (!good_keypoint_distribution(cur_frame)) return std::nullopt;
      RCLCPP_INFO(logger_, "found good reference frame");
      ref_frame_ = std::move(cur_frame);
      state_ = State::INITIALIZING;
      return std::nullopt;
    }

    if (state_ == State::INITIALIZING) {
      std::vector<std::vector<cv::DMatch>> knn_matches;
      auto ref_descriptors = ref_frame_.get_descriptors();
      auto cur_descriptors = cur_frame.get_descriptors();

      std::vector<cv::DMatch> good_matches =
        feature_extractor_->find_matches(ref_descriptors, cur_descriptors, lowes_distance_ratio_);

      if (good_matches.size() < min_matches_for_parallax_) {
        RCLCPP_WARN(logger_, "Initializer: Not enough matches");
        // check if new frame is good ref, if yes set it else continue initializing step
        if (good_keypoint_distribution(cur_frame)) {
          RCLCPP_WARN(logger_, "found new good reference frame");
          ref_frame_ = std::move(cur_frame);
        } else {
          RCLCPP_WARN(logger_, "Resetting state");
          reset();
        }
        return std::nullopt;
      }

      std::vector<Observation> new_ref_observations, new_cur_observations;
      for (const auto & match : good_matches) {
        new_ref_observations.push_back(ref_frame_.observations[match.queryIdx]);
        new_cur_observations.push_back(cur_frame.observations[match.trainIdx]);
      }
      ref_frame_.observations = std::move(new_ref_observations);
      cur_frame.observations = std::move(new_cur_observations);

      cv::Mat img_matches = utils::draw_matched_frames(ref_frame_, cur_frame);
      cv::imshow("Matches", img_matches);
      cv::waitKey(1);

      std::vector<cv::Point2f> pts_ref = ref_frame_.get_points_2d();
      std::vector<cv::Point2f> pts_cur = cur_frame.get_points_2d();
      if (!check_parallax(pts_ref, pts_cur)) {
        RCLCPP_WARN(logger_, "Parallax check failed");
        return std::nullopt;
      }

      RCLCPP_INFO(logger_, "Parallax check passed");

      // find essential matrix
      std::vector<uchar> inlier_mask;
      cv::Mat E = cv::findEssentialMat(pts_ref, pts_cur, K, cv::RANSAC, 0.99, 1.0, inlier_mask);
      RCLCPP_INFO(
        logger_, "Found Essential Matrix with inlier ratio %lf",
        static_cast<double>(cv::countNonZero(inlier_mask)) / pts_ref.size());

      // decompose to get rotation and translation
      cv::Mat R_cw, t_cw;
      cv::recoverPose(E, pts_ref, pts_cur, K, R_cw, t_cw, inlier_mask);

      uint32_t num_inliers = cv::countNonZero(inlier_mask);

      // print R,t and inliers
      RCLCPP_INFO(
        logger_, "Inlier ratio after pose recovery: %lf",
        static_cast<double>(num_inliers) / pts_ref.size());

      // filter points
      ref_frame_.filter_observations_by_mask(inlier_mask);
      cur_frame.filter_observations_by_mask(inlier_mask);

      // triangulate points

      std::vector<uchar> inliers;
      std::vector<cv::Point3f> pts3d = traingulate_points(
        K, R_cw, t_cw, ref_frame_.get_points_2d(), cur_frame.get_points_2d(), inliers);

      // filter chirality check passed points
      ref_frame_.filter_observations_by_mask(inliers);  // only used for vis after this
      cur_frame.filter_observations_by_mask(inliers);

      // check min 4 points are valid for PnP later
      if (pts3d.size() < 4) {
        RCLCPP_WARN(logger_, "Less than 4 points triangulated: resetting initializer");
        reset();
        return std::nullopt;
      }

      // add origin keyframe
      KeyFrame::Ptr origin_keyframe =
        std::make_shared<KeyFrame>(cv::Affine3d(cv::Matx33d::eye(), cv::Vec3d::zeros()));
      map_->add_keyframe(origin_keyframe);

      // set frame pose
      cur_frame.pose_wc = cv::Affine3d(R_cw, t_cw).inv();

      // filter based on new 3D landmarks
      for (size_t i = 0; i < pts3d.size(); ++i) {
        Landmark lm = Landmark{pts3d[i], cur_frame.observations[i].descriptor};
        map_->add_landmark(lm);
        cur_frame.observations[i].landmark_id = lm.id;
      }

      map_->add_keyframe(std::make_shared<KeyFrame>(cur_frame));

      img_matches = utils::draw_matched_frames(ref_frame_, cur_frame);
      cv::imshow("Matches", img_matches);
      cv::waitKey(1);

      ref_frame_ = cur_frame;
      state_ = State::INITIALIZED;
      return ref_frame_;
    }
    return std::nullopt;
  }

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

  FeatureExtractor::Ptr feature_extractor_;
};
}  // namespace mono_vo