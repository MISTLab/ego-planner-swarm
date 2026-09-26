#ifndef _GRID_MAP_H
#define _GRID_MAP_H

#include <Eigen/Eigen>
#include <Eigen/StdVector>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <plan_env/raycast.h>
#include <plan_env/rolling_window.h>

using namespace std;

/*
 * SwarmDeck's rolling occupancy map. Upstream allocated dense arrays for the
 * whole mission box (about 15 bytes per cell, fixed origin: roughly 9 GB for
 * the SubT world at 0.1 m). Here the arrays cover a window centred on the
 * drone (24 x 24 x 8 m at 0.1 m by default, about 11 bytes per cell), indexed
 * modulo the window (RollingWindow), whatever the mission size. Voxels
 * leaving the window are forgotten; everything outside it reads as blocked.
 * Input is the lidar cloud in the odometry frame with its sensor origin; only
 * hits inside the window are fused; outside returns carve misses to the edge
 * or ray limit, by ray casting into float log-odds.
 */

struct MappingParameters
{
  Eigen::Vector3d window_size_;
  double resolution_;
  double obstacles_inflation_;
  string frame_id_;
  double p_hit_, p_miss_, p_min_, p_max_, p_occ_;
  float prob_hit_log_, prob_miss_log_, clamp_min_log_, clamp_max_log_, min_occupancy_log_;
  float unknown_flag_;
  double max_ray_length_;
  double virtual_ceil_height_;
  double visualization_truncate_height_;
  double sensor_timeout_;
};

struct MappingData
{
  vector<float> occupancy_buffer_;
  vector<char> occupancy_buffer_inflate_;
  vector<short> count_hit_, count_hit_and_miss_;
  vector<char> flag_traverse_, flag_rayend_;
  char raycast_num_;
  queue<Eigen::Vector3i> cache_voxel_;
  vector<Eigen::Vector3d> points_;
  Eigen::Vector3d sensor_pos_;
  Eigen::Vector3i local_bound_min_, local_bound_max_;
  bool has_odom_, has_cloud_, local_updated_, flag_sensor_timeout_;
  std::chrono::steady_clock::time_point last_cloud_;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class GridMap
{
public:
  typedef std::shared_ptr<GridMap> Ptr;
  using OccupancyValue = float;

  enum
  {
    INVALID_IDX = -10000
  };

  GridMap() {}
  ~GridMap() {}

  void initMap(rclcpp::Node::SharedPtr node);

  /** Centre the window on the drone; voxels leaving it are forgotten. */
  void recenter(const Eigen::Vector3d &pos);
  /** Fuse one scan: points in the odometry frame, cast from `sensor_origin`. */
  void inputCloud(const vector<Eigen::Vector3d> &points, const Eigen::Vector3d &sensor_origin);

  std::size_t bufferCells() const { return window_.cells(); }
  Eigen::Vector3d windowMin() const { return window_.minBound(); }
  Eigen::Vector3d windowMax() const { return window_.maxBound(); }

  inline int getInflateOccupancy(Eigen::Vector3d pos);
  inline int getOccupancy(Eigen::Vector3d pos);
  inline bool isInMap(const Eigen::Vector3d &pos);
  inline bool isInMap(const Eigen::Vector3i &idx);
  inline bool isUnknown(const Eigen::Vector3d &pos);
  inline void posToIndex(const Eigen::Vector3d &pos, Eigen::Vector3i &id);
  inline void indexToPos(const Eigen::Vector3i &id, Eigen::Vector3d &pos);
  inline double getResolution();

  bool odomValid() { return md_.has_odom_; }
  /** True while no lidar cloud has arrived for grid_map/odom_depth_timeout s */
  bool getOdomDepthTimeout() { return md_.flag_sensor_timeout_; }

  void publishMap();
  void publishMapInflate(bool all_info = false);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  typedef message_filters::sync_policies::ExactTime<sensor_msgs::msg::PointCloud2,
                                                    geometry_msgs::msg::PointStamped>
      SyncPolicyCloudOrigin;
  typedef shared_ptr<message_filters::Synchronizer<SyncPolicyCloudOrigin>> SynchronizerCloudOrigin;

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr odom);
  void cloudOriginCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &cloud,
                           const geometry_msgs::msg::PointStamped::ConstSharedPtr &origin);
  void checkSensorTimeout();
  void resetBox(const Eigen::Vector3i &min_id, const Eigen::Vector3i &max_id);
  int setCacheOccupancy(const Eigen::Vector3d &pos, int occ);
  void raycastProcess();
  void clearAndInflateLocalMap();
  void inflateBox(const Eigen::Vector3i &min_id, const Eigen::Vector3i &max_id);

  MappingParameters mp_;
  MappingData md_;
  RollingWindow window_;

  rclcpp::Node::SharedPtr node_;
  shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> cloud_sub_;
  shared_ptr<message_filters::Subscriber<geometry_msgs::msg::PointStamped>> origin_sub_;
  SynchronizerCloudOrigin sync_cloud_origin_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_inf_pub_;
  rclcpp::TimerBase::SharedPtr timeout_timer_;
  rclcpp::TimerBase::SharedPtr vis_timer_;
};

inline int GridMap::getInflateOccupancy(Eigen::Vector3d pos)
{
  const Eigen::Vector3i id = window_.indexOf(pos);
  if (!window_.contains(id))
    return -1;
  return int(md_.occupancy_buffer_inflate_[window_.address(id)]);
}

inline int GridMap::getOccupancy(Eigen::Vector3d pos)
{
  const Eigen::Vector3i id = window_.indexOf(pos);
  if (!window_.contains(id))
    return -1;
  return md_.occupancy_buffer_[window_.address(id)] > mp_.min_occupancy_log_ ? 1 : 0;
}

inline bool GridMap::isInMap(const Eigen::Vector3d &pos) { return window_.contains(pos); }

inline bool GridMap::isInMap(const Eigen::Vector3i &idx) { return window_.contains(idx); }

inline bool GridMap::isUnknown(const Eigen::Vector3d &pos)
{
  const Eigen::Vector3i id = window_.indexOf(pos);
  if (!window_.contains(id))
    return true;
  return md_.occupancy_buffer_[window_.address(id)] < mp_.clamp_min_log_ - 1e-3f;
}

inline void GridMap::posToIndex(const Eigen::Vector3d &pos, Eigen::Vector3i &id)
{
  id = window_.indexOf(pos);
}

inline void GridMap::indexToPos(const Eigen::Vector3i &id, Eigen::Vector3d &pos)
{
  pos = window_.centerOf(id);
}

inline double GridMap::getResolution() { return mp_.resolution_; }

#endif
