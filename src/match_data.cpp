#include "mono_vo/match_data.hpp"

namespace mono_vo
{
std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> extract_points_from_matches(
  const std::vector<MatchData> & matches)
{
  std::vector<cv::Point2f> pts1, pts2;
  pts1.reserve(matches.size());
  pts2.reserve(matches.size());
  for (const auto & match : matches) {
    pts1.push_back(match.pt_ref);
    pts2.push_back(match.pt_cur);
  }
  return std::make_pair(pts1, pts2);
}

}  // namespace mono_vo