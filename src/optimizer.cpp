#pragma once

#include "mono_vo/optimizer.hpp"

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/types/sba/types_sba.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/se3_ops.h>

#include <set>
#include <sophus/se3.hpp>

#include "mono_vo/map.hpp"

namespace mono_vo
{

g2o::SE3Quat to_g2o(const Sophus::SE3d & pose)
{
  return g2o::SE3Quat(pose.rotationMatrix(), pose.translation());
}

Sophus::SE3d to_sophus(const g2o::SE3Quat & se3)
{
  return Sophus::SE3d(se3.rotation(), se3.translation());
}

Optimizer::Optimizer(const cv::Mat K, rclcpp::Logger logger = rclcpp::get_logger("Optimizer"))
: K_(K), logger_(logger)
{
  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);
  cam_params_ = new g2o::CameraParameters(fx, Eigen::Vector2d(cx, cy), 0);
}

void Optimizer::local_bundle_adjustment(Map::Ptr map, int local_window_size)
{
  // --- Setup optimizer
  g2o::SparseOptimizer optimizer;
  auto linear_solver =
    std::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();
  auto block_solver = std::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
  auto * solver = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
  optimizer.setAlgorithm(solver);
  // optimizer.setVerbose(true); for debug

  // --- identify local window of keyframes and landmarks
  std::vector<KeyFrame::Ptr> local_keyframes;
  std::set<long> local_landmark_ids;
  long max_kf_id = 0;  // offset for landmark ids

  // get last local window size keyframes
  auto all_keyframes = map->get_all_keyframes();
  for (auto it = all_keyframes.rbegin();
       it != all_keyframes.rend() && local_keyframes.size() < local_window_size; it++) {
    local_keyframes.push_back(it->second);
    if (it->first > max_kf_id) max_kf_id = it->first;
  }

  // get all unique landmarks seen by these keyframes
  for (const auto & kf : local_keyframes) {
    for (const auto & obs : kf->observations) {
      if (obs.landmark_id != -1) {
        local_landmark_ids.insert(obs.landmark_id);
      }
    }
  }

  RCLCPP_INFO(
    logger_, "Local BA on %zu KeyFrames and %zu Landmarks.", local_keyframes.size(),
    local_landmark_ids.size());
  if (local_keyframes.empty() || local_landmark_ids.empty()) {
    RCLCPP_WARN(logger_, "Not enough data for Local BA, skipping.");
    return;
  }

  // --- create graph from data
  cam_params_->setId(0);
  optimizer.addParameter(cam_params_);

  std::map<long, g2o::VertexSE3Expmap *> pose_vertices;
  std::map<long, g2o::VertexPointXYZ *> landmark_vertices;

  // add keyframe pose vertices
  for (const auto & kf : local_keyframes) {
    auto * v_pose = new g2o::VertexSE3Expmap();
    v_pose->setId(kf->id);
    v_pose->setEstimate(to_g2o(kf->pose_wc));
    // fix the origin or last keyframe
    v_pose->setFixed(kf == local_keyframes.back());
    optimizer.addVertex(v_pose);
    pose_vertices[kf->id] = v_pose;
  }

  // add landmark point vertices
  const long lm_id_offset = max_kf_id + 1;
  for (const long lm_id : local_landmark_ids) {
    const auto & lm = map->get_landmark(lm_id);
    auto * v_point = new g2o::VertexPointXYZ();
    v_point->setId(lm_id + lm_id_offset);
    v_point->setEstimate(g2o::Vector3(lm.pose_w.x, lm.pose_w.y, lm.pose_w.z));
    v_point->setMarginalized(true);
    optimizer.addVertex(v_point);
    landmark_vertices[lm_id] = v_point;
  }

  // --- add edges (measurements)
  const float chi2_th_sqrt = std::sqrt(5.991);  // 95% confidence for 2 DoF (u, v)
  for (const auto & kf : local_keyframes) {
    for (const auto & obs : kf->observations) {
      // Ensure the observed landmark is part of local optimization
      if (local_landmark_ids.find(obs.landmark_id) == local_landmark_ids.end()) continue;

      auto * edge = new g2o::EdgeProjectXYZ2UV();
      edge->setVertex(0, landmark_vertices.at(obs.landmark_id + lm_id_offset));
      edge->setVertex(1, pose_vertices.at(kf->id));

      edge->setMeasurement(g2o::Vector2(obs.keypoint.pt.x, obs.keypoint.pt.y));
      const float inv_sigma2 = 1.0f / (1 << obs.keypoint.octave);
      edge->setInformation(Eigen::Matrix2d::Identity() * inv_sigma2);  // noise
      edge->setRobustKernel(new g2o::RobustKernelHuber());
      edge->robustKernel()->setDelta(chi2_th_sqrt);  // pixels
      edge->setParameterId(0, 0);                    // camera params with id 0
      optimizer.addEdge(edge);
    }
  }

  // --- optimize
  optimizer.initializeOptimization();
  optimizer.setVerbose(true);  // for debug
  optimizer.optimize(10);

  RCLCPP_INFO(logger_, "Optimization finished. Updating map...");

  // --- update keyframes and landmarks

  // update keyframe poses
  for (auto const & [kf_id, v_pose] : pose_vertices) {
    const auto & kf = map->get_keyframe(kf_id);
    if (kf) {
      kf->pose_wc = to_sophus(v_pose->estimate());
    }
  }

  // update landmarks
  for (auto const & [lm_id, v_point] : landmark_vertices) {
    auto & lm = map->get_landmark_ref(lm_id);
    lm.pose_w = cv::Point3f(v_point->estimate()[0], v_point->estimate()[1], v_point->estimate()[2]);
  }

  RCLCPP_INFO(logger_, "Local map updated after BA.");
}

}  // namespace mono_vo