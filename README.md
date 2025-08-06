# ros2_mono_vo

A simple monocular visual odometry system for ROS2. This package provides a single node that estimates the 6-DOF pose of a camera from a monocular image stream.

## Prerequisites

- ROS 2 (Tested on Jazzy)
- OpenCV
- A calibrated camera (undistorted) publishing `sensor_msgs/Image` and `sensor_msgs/CameraInfo` messages.

## How to Launch

1.  Build the package:
    ```bash
    colcon build --packages-up-to mono_vo
    ```

2.  Source the workspace:
    ```bash
    source install/setup.bash
    ```

3.  Run the launch file. This will start the VO node and RViz for visualization.
    ```bash
    ros2 launch mono_vo mono_vo.launch.py
    ```

## Subscribed Topics

| Topic | Type | Description |
| :--- | :--- | :--- |
| `/camera/image_rect` | `sensor_msgs::msg::Image` | Rectified camera image used for tracking features. |
| `/camera/camera_info` | `sensor_msgs::msg::CameraInfo` | Camera intrinsic parameters required for projection and triangulation. |

## Published Topics

| Topic | Type | Description |
| :--- | :--- | :--- |
| `/camera/pose` | `geometry_msgs::msg::PoseStamped` | The estimated 6-DOF pose of the camera in the odometry frame. |
| `/camera/path` | `nav_msgs::msg::Path` | The full estimated trajectory of the camera, useful for visualization in RViz. |
| `/camera/pointcloud` | `sensor_msgs::msg::PointCloud2` | The current 3D map points triangulated from keyframes. |

## Parameters

All parameters are loaded from the `config/vo_params.yaml` file at launch.

### Initializer Parameters

These parameters control the two-view initialization process.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `initializer.occupancy_grid_div` | `int` | `50` | Grid size (N) for checking keypoint distribution. |
| `initializer.kp_distribution_thresh`| `double` | `0.5` | Minimum ratio of grid cells that must contain keypoints. |
| `initializer.lowes_distance_ratio` | `double` | `0.7` | Lowe's ratio for feature matching; lower is stricter. |
| `initializer.min_matches_for_init` | `int` | `100` | Minimum matches needed to attempt VO initialization. |
| `initializer.ransac_reproj_thresh` | `double` | `1.0` | RANSAC reprojection threshold (pixels) for H/F model fitting. |
| `initializer.f_inlier_thresh` | `double` | `0.5` | Minimum inlier ratio to accept the Fundamental matrix model. |
| `initializer.model_score_thresh` | `double` | `0.56`| Score ratio (H/F) threshold to decide between planar and general scene models. |

### Tracker Parameters

These parameters control the frame-to-frame tracking and keyframe generation logic.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `tracker.tracking_error_thresh` | `double` | `30.0` | Lucas-Kanade (LK) optical flow tracking error threshold in pixels. |
| `tracker.min_observations_before_triangulation` | `int` | `100` | Keyframe trigger: minimum total feature observations required. |
| `tracker.min_tracked_points` | `int` | `10` | Minimum tracked points required; tracking is declared lost if count falls below this. |
| `tracker.max_tracking_after_keyframe` | `int` | `10` | Keyframe trigger: maximum number of frames to track after the last keyframe. |
| `tracker.max_rotation_from_keyframe` | `double` | `0.2618` | Keyframe trigger: maximum rotation from the last keyframe (radians, 15 deg). |
| `tracker.max_translation_from_keyframe` | `double` | `1.0` | Keyframe trigger: maximum translation (meters) from the last keyframe. |
| `tracker.ransac_reproj_thresh` | `double` | `1.0` | RANSAC reprojection threshold (pixels) for estimating pose. |
| `tracker.model_score_thresh` | `double` | `0.85` | H/F model score threshold for robust pose estimation. |
| `tracker.f_inlier_thresh` | `double` | `0.5` | Minimum inlier ratio to accept the Fundamental matrix model for pose estimation. |
| `tracker.lowes_distance_ratio` | `double` | `0.7` | Lowe's ratio for matching new features against the last keyframe. |

## Algorithm

### Initializer

![initializer_algorithm_flowchart](./images/mono_vo_initializer.png)

### Tracker

![tracker_algorithm_flowchart](./images/mono_vo_tracker.png)