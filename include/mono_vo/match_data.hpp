#pragma once

#include <opencv2/core.hpp>
#include <utility>
#include <vector>

namespace mono_vo
{
struct MatchData
{
  MatchData(int ref_idx, int cur_idx, const cv::Point2f & pt_ref, const cv::Point2f & pt_cur)
  : ref_idx(ref_idx), cur_idx(cur_idx), pt_ref(pt_ref), pt_cur(pt_cur)
  {
  }

  int ref_idx;         // Original index in ref_frame_.observations
  int cur_idx;         // Original index in cur_frame.observations
  cv::Point2f pt_ref;  // 2D point in reference frame
  cv::Point2f pt_cur;  // 2D point in current frame
};

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> extract_points_from_matches(
  const std::vector<MatchData> & matches);

}  // namespace mono_vo