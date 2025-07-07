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

  bool try_initializing(const cv::Mat & grey_img, const cv::Matx33d & K)
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