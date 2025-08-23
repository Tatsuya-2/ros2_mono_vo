#pragma once

#include <g2o/types/sba/types_six_dof_expmap.h>

#include <memory>

#include "mono_vo/map.hpp"

namespace mono_vo
{
class Optimizer
{
public:
  using Ptr = std::shared_ptr<Optimizer>;

  Optimizer(rclcpp::Logger = rclcpp::get_logger("Optimizer"));

  void set_camera_params(const cv::Mat & K);

  /**
   * @brief Performs local bundle adjustment on a window of recent keyframes.
   * @param map The map containing all keyframes and landmarks.
   * @param local_window_size The number of recent keyframes to include in the optimization.
   */
  void local_bundle_adjustment(Map::Ptr map, size_t local_window_size = 10);

private:
  cv::Mat K_;
  rclcpp::Logger logger_;
  g2o::CameraParameters * cam_params_;
};
}  // namespace mono_vo