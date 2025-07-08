#pragma once

#include <opencv2/opencv.hpp>
#include <optional>
#include <rclcpp/logging.hpp>
#include <vector>

namespace mono_vo
{
struct FrameData
{
  cv::Mat image;
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
};

class Initializer
{
public:
  enum class State
  {
    OBTAINING_REF,
    INITIALIZING,
    INITIALIZED
  };

  Initializer(rclcpp::Logger logger = rclcpp::get_logger("Initializer"))
  : logger_(logger),
    state_(State::OBTAINING_REF),
    distribution_thresh_(0.5f),
    matcher_(cv::BFMatcher(cv::NORM_HAMMING))
  {
    orb_det_ = cv::ORB::create(1000);
  }

  bool is_initalized() { return state_ == State::INITIALIZED; }

  void reset() { state_ = State::OBTAINING_REF; }

  bool good_ref_frame(const FrameData & frame)
  {
    RCLCPP_INFO(logger_, "totals kps: %ld", frame.keypoints.size());

    // check distribution in a grid across the image
    cv::Mat grid = cv::Mat::zeros(frame.image.rows / 50, frame.image.cols / 50, CV_8U);
    int occupied_cells = 0;
    for (auto & kp : frame.keypoints) {
      int r = kp.pt.y / 50, c = kp.pt.x / 50;
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

    // check 2: Is the Fundamental Matrix a significantly better model?
    // The ratio score_H / score_F should be low.
    // A high ratio means Homography explains the data almost as well as Fundamental Matrix.
    if (model_score > model_score_thresh_) {
      return false;
    }

    return true;
  }

  bool try_initializing(const cv::Mat & grey_img, const cv::Mat & K)
  {
    FrameData cur_frame;
    cur_frame.image = grey_img;
    orb_det_->detectAndCompute(
      cur_frame.image, cv::Mat(), cur_frame.keypoints, cur_frame.descriptors);

    if (state_ == State::OBTAINING_REF) {
      if (!good_ref_frame(cur_frame)) return false;
      RCLCPP_INFO(logger_, "found good reference frame");
      ref_frame_ = std::move(cur_frame);
      state_ = State::INITIALIZING;
      return false;
    }

    if (state_ == State::INITIALIZING) {
      std::vector<std::vector<cv::DMatch>> knn_matches;
      matcher_.knnMatch(ref_frame_.descriptors, cur_frame.descriptors, knn_matches, 2);
      RCLCPP_INFO(logger_, "total matches: %ld", knn_matches.size());

      std::vector<cv::DMatch> good_matches;
      for (auto & match : knn_matches) {
        if (match.size() == 2 && match[0].distance < 0.7 * match[1].distance) {
          good_matches.push_back(match[0]);
        }
      }

      RCLCPP_INFO(logger_, "good matches: %ld", good_matches.size());

      if (good_matches.size() < min_matches_for_parallax_) {
        RCLCPP_WARN(logger_, "Initializer: Not enough matches");
        // check if new frame is good ref, if yes set it else continue initializing step
        if (good_ref_frame(cur_frame)) {
          RCLCPP_WARN(logger_, "found new good reference frame");
          ref_frame_ = std::move(cur_frame);
        } else {
          RCLCPP_WARN(logger_, "Resetting state");
          reset();
        }
        return false;
      }

      cv::Mat img_matches;
      cv::drawMatches(
        ref_frame_.image, ref_frame_.keypoints, cur_frame.image, cur_frame.keypoints, good_matches,
        img_matches);

      std::vector<cv::Point2f> pts1, pts2;
      for (const auto & match : good_matches) {
        pts1.push_back(ref_frame_.keypoints[match.queryIdx].pt);
        pts2.push_back(cur_frame.keypoints[match.trainIdx].pt);
      }

      if (!check_parallax(pts1, pts2)) {
        RCLCPP_WARN(logger_, "Parallax check failed");
        cv::imshow("Matches", img_matches);
        cv::waitKey(1);
        return false;
      }

      RCLCPP_INFO(logger_, "Parallax check passed");

      // find essential matrix
      std::vector<uchar> inlier_mask;
      cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.99, 1.0, inlier_mask);
      RCLCPP_INFO(
        logger_, "Found Essential Matrix with inlier ratio %lf",
        static_cast<double>(cv::countNonZero(inlier_mask)) / pts1.size());

      // decompose to get rotation and translation
      cv::Mat R, t;
      cv::recoverPose(E, pts1, pts2, K, R, t, inlier_mask);

      // print R,t and inliers
      std::cout << "R:\n" << R << std::endl;
      std::cout << "t:\n" << t << std::endl;
      std::cout << "Inlier  ratio:\n"
                << static_cast<double>(cv::countNonZero(inlier_mask)) / pts1.size() << std::endl;

      // get projection matrices
      cv::Mat P_ref = cv::Mat::zeros(3, 4, CV_64F);
      K.copyTo(P_ref(cv::Rect(0, 0, 3, 3)));  // P_ref = K * [I | 0]

      cv::Mat P_cur = cv::Mat::zeros(3, 4, CV_64F);
      R.copyTo(P_cur(cv::Rect(0, 0, 3, 3)));
      t.copyTo(P_cur(cv::Rect(3, 0, 1, 3)));
      P_cur = K * P_cur;  // P_cur = K * [R | t]

      // filter points
      std::vector<cv::Point2f> inlier_pts1;
      std::vector<cv::Point2f> inlier_pts2;
      for (int i = 0; i < inlier_mask.size(); ++i) {
        if (inlier_mask[i]) {
          inlier_pts1.push_back(pts1[i]);
          inlier_pts2.push_back(pts2[i]);
        }
      }

      // triangulate points
      cv::Mat pts4d_h;  // Output is 4xN matrix of homogeneous 3D points
      cv::triangulatePoints(P_ref, P_cur, inlier_pts1, inlier_pts2, pts4d_h);

      RCLCPP_INFO(logger_, "Triangulated %d 3D points", pts4d_h.cols);

      // convert to cartesian
      cv::Mat pts3d;
      cv::convertPointsFromHomogeneous(pts4d_h.t(), pts3d);

      // chirality check
      // The 3D points are in the reference camera's coordinate system.
      // We need to keep only the points that are:
      // 1. In front of the reference camera (z > 0).
      // 2. In front of the current camera (z > 0 after transformation).

      std::vector<cv::Point3f> valid_3d_points;
      std::vector<cv::Point2f> valid_keypoints_ref;     // To keep track of the original 2D points
      std::vector<cv::Point2f> valid_keypoints_cur;     // that correspond to our valid 3D points
      std::vector<bool> point_mask(pts3d.rows, false);  // A mask to track valid points

      for (int i = 0; i < pts3d.rows; ++i) {
        // The result of convertPointsFromHomogeneous is a Mat of CV_32FC3 (3-channel float)
        // or a Mat of cv::Point3f. We'll access it as Point3f.
        const cv::Point3f & p3d_ref = pts3d.at<cv::Point3f>(i);

        // --- Check 1: Point is in front of the reference camera ---
        // The point is already in the reference camera's frame.
        if (p3d_ref.z <= 0) {
          continue;  // Point is behind the first camera, discard.
        }

        // --- Check 2: Point is in front of the current camera ---
        // Transform the 3D point into the current camera's coordinate system.
        // The pose (R, t) transforms points from the current frame to the reference frame.
        // p_ref = R * p_cur + t
        // We need the inverse to get p_cur. Or simpler, `recoverPose` gives (R,t) such that
        // a point X in the world (ref frame) is seen as R*X+t in the current camera's frame.
        cv::Mat p3d_ref_mat = (cv::Mat_<double>(3, 1) << p3d_ref.x, p3d_ref.y, p3d_ref.z);
        cv::Mat p3d_cur_mat = R * p3d_ref_mat + t;

        if (p3d_cur_mat.at<double>(2, 0) <= 0) {
          continue;  // Point is behind the second camera, discard.
        }

        // If both checks pass, this is a valid point.
        point_mask[i] = true;
        valid_3d_points.push_back(p3d_ref);

        // Also save the corresponding 2D keypoints. This is very useful for tracking.
        // NOTE: You must use the original `pts1` and `pts2` from BEFORE you filtered them
        // with the essential matrix mask (`E_mask`), or apply the same mask here.
        // Assuming `pts1` and `pts2` here are the RANSAC inliers for E.
        valid_keypoints_ref.push_back(inlier_pts1[i]);
        valid_keypoints_cur.push_back(inlier_pts2[i]);
      }

      RCLCPP_INFO(
        logger_, "Chirality check complete. Kept %zu / %d valid points.", valid_3d_points.size(),
        pts3d.rows);

      cv::imshow("Matches", img_matches);
      cv::waitKey(0);

      state_ = State::INITIALIZED;
      return true;
    }
  }

private:
  rclcpp::Logger logger_;
  State state_;
  FrameData ref_frame_;
  float distribution_thresh_;
  double min_matches_for_parallax_ = 100;
  double ransac_thresh_h_ = 2.0;     // px homography RANSAC threshold
  double ransac_thresh_f_ = 1.0;     // px fundamental RANSAC threshold
  double f_inlier_thresh_ = 0.5;     // fundamental inlier threshold ratio
  double model_score_thresh_ = 0.5;  // max H/F ratio

  cv::Ptr<cv::ORB> orb_det_;
  cv::BFMatcher matcher_;
};
}  // namespace mono_vo