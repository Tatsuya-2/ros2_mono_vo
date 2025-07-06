#pragma once

#include <opencv2/opencv.hpp>
#include <optional>
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
  Initializer()
  : initalized_(false), distribution_thresh_(0.5f), matcher_(cv::BFMatcher(cv::NORM_HAMMING))
  {
    orb_det_ = cv::ORB::create(1000);
  }

  bool is_initalized() { return initalized_; }

  std::optional<FrameData> good_first_frame(const cv::Mat & grey_img)
  {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb_det_->detectAndCompute(grey_img, cv::Mat(), keypoints, descriptors);

    std::cout << "totals kps: " << keypoints.size() << std::endl;

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
    std::cout << "occupied cells: " << occupied_cells << " total cells: " << total_cells
              << std::endl;
    auto occupancy = static_cast<float>(occupied_cells) / total_cells;
    std::cout << "occupancy: " << occupancy << std::endl;
    if (occupancy > distribution_thresh_) {
      return FrameData{grey_img, keypoints, descriptors};
    }
    return std::nullopt;
  }

  void update(const cv::Mat & grey_img)
  {
    if (!first_frame_.has_value()) {
      first_frame_ = good_first_frame(grey_img);
      if (!first_frame_.has_value()) return;
      std::cout << "found good first frame" << std::endl;
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb_det_->detectAndCompute(grey_img, cv::Mat(), keypoints, descriptors);

    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_.knnMatch(first_frame_->descriptors, descriptors, knn_matches, 2);
    std::cout << "total matches: " << knn_matches.size() << std::endl;

    std::vector<cv::DMatch> good_matches;
    for (auto & match : knn_matches) {
      if (match.size() == 2 && match[0].distance < 0.7 * match[1].distance) {
        good_matches.push_back(match[0]);
      }
    }

    std::cout << "good matches: " << good_matches.size() << std::endl;

    cv::Mat img_matches;
    cv::drawMatches(
      first_frame_->image, first_frame_->keypoints, grey_img, keypoints, good_matches, img_matches);

    cv::imshow("Matches", img_matches);
    cv::waitKey(0);
  }

private:
  bool initalized_;
  std::optional<FrameData> first_frame_;
  float distribution_thresh_;

  cv::Ptr<cv::ORB> orb_det_;
  cv::BFMatcher matcher_;
};
}  // namespace mono_vo