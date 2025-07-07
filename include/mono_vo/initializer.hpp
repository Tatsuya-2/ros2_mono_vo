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
  Initializer(rclcpp::Logger logger = rclcpp::get_logger("Initializer"))
  : logger_(logger),
    initalized_(false),
    distribution_thresh_(0.5f),
    matcher_(cv::BFMatcher(cv::NORM_HAMMING))
  {
    orb_det_ = cv::ORB::create(1000);
  }

  bool is_initalized() { return initalized_; }

  std::optional<FrameData> good_first_frame(const cv::Mat & grey_img)
  {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb_det_->detectAndCompute(grey_img, cv::Mat(), keypoints, descriptors);

    RCLCPP_INFO(logger_, "totals kps: %ld", keypoints.size());

    // check distribution in a grid across the image
    cv::Mat grid = cv::Mat::zeros(grey_img.rows / 50, grey_img.cols / 50, CV_8U);
    int occupied_cells = 0;
    for (auto & kp : keypoints) {
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
      return FrameData{grey_img, keypoints, descriptors};
    }
    return std::nullopt;
  }

  bool check_parallax(const std::vector<cv::Point2f> & pts1, const std::vector<cv::Point2f> & pts2)
  {
    if (pts1.size() != pts2.size() || pts1.size() < 100) return false;

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

  void update(const cv::Mat & grey_img, const cv::Matx33d & K)
  {
    if (!first_frame_.has_value()) {
      first_frame_ = good_first_frame(grey_img);
      if (!first_frame_.has_value()) return;
      RCLCPP_INFO(logger_, "found good first frame");
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb_det_->detectAndCompute(grey_img, cv::Mat(), keypoints, descriptors);

    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_.knnMatch(first_frame_->descriptors, descriptors, knn_matches, 2);
    RCLCPP_INFO(logger_, "total matches: %ld", knn_matches.size());

    std::vector<cv::DMatch> good_matches;
    for (auto & match : knn_matches) {
      if (match.size() == 2 && match[0].distance < 0.7 * match[1].distance) {
        good_matches.push_back(match[0]);
      }
    }

    RCLCPP_INFO(logger_, "good matches: %ld", good_matches.size());

    cv::Mat img_matches;
    cv::drawMatches(
      first_frame_->image, first_frame_->keypoints, grey_img, keypoints, good_matches, img_matches);

    std::vector<cv::Point2f> pts1, pts2;
    for (const auto & match : good_matches) {
      pts1.push_back(first_frame_->keypoints[match.queryIdx].pt);
      pts2.push_back(keypoints[match.trainIdx].pt);
    }

    if (!check_parallax(pts1, pts2)) {
      RCLCPP_WARN(logger_, "Parallax check failed");
      cv::imshow("Matches", img_matches);
      cv::waitKey(1);
      return;
    }

    RCLCPP_INFO(logger_, "Parallax check passed");
    cv::imshow("Matches", img_matches);
    cv::waitKey(0);

    // RCLCPP_INFO(logger_, " K: " << K);

    // cv::Mat mask;
    // cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, 1.0, mask);

    // if (E.empty()) {
    //   RCLCPP_INFO(logger_, "Essential matrix estimation failed");
    //   return;
    // }
    // RCLCPP_INFO(logger_, "Essential matrix estimated\n" << E);

    // cv::Mat R, t;
    // int inliers = cv::recoverPose(E, pts1, pts2, K, R, t, mask);

    // RCLCPP_INFO(logger_, "inliers: " << inliers);

    // // Check inlier ratio
    // double inlier_ratio = static_cast<double>(inliers) / pts1.size();
    // if (inlier_ratio < 0.3) {
    //   RCLCPP_INFO(logger_, "Too few inliers: " << inlier_ratio);
    //   return;
    // }

    // RCLCPP_INFO(logger_, "Initial pose estimated");

    // // Triangulate 3D points
    // cv::Mat P1 = K * cv::Mat::eye(3, 4, CV_64F);  // First camera [I|0]
    // cv::Mat P2 =
    //   K * (cv::Mat_<double>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
    //        t.at<double>(0, 0), R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
    //        t.at<double>(1, 0), R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2),
    //        t.at<double>(2, 0));

    // cv::Mat points4D;
    // cv::triangulatePoints(P1, P2, pts1, pts2, points4D);

    // RCLCPP_INFO(logger_, "3D points estimated");
  }

private:
  rclcpp::Logger logger_;
  bool initalized_;
  std::optional<FrameData> first_frame_;
  float distribution_thresh_;
  double ransac_thresh_h_ = 2.0;      // px homography RANSAC threshold
  double ransac_thresh_f_ = 1.0;      // px fundamental RANSAC threshold
  double f_inlier_thresh_ = 0.5;      // fundamental inlier threshold ratio
  double model_score_thresh_ = 0.45;  // max H/F ratio

  cv::Ptr<cv::ORB> orb_det_;
  cv::BFMatcher matcher_;
};
}  // namespace mono_vo