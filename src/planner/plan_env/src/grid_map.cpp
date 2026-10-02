#include "plan_env/grid_map.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{
  float logOdds(double probability)
  {
    return static_cast<float>(std::log(probability / (1.0 - probability)));
  }
} // namespace

void GridMap::initMap(rclcpp::Node::SharedPtr node)
{
  node_ = node;

  node_->declare_parameter("grid_map/resolution", 0.1);
  node_->declare_parameter("grid_map/window_size_x", 24.0);
  node_->declare_parameter("grid_map/window_size_y", 24.0);
  node_->declare_parameter("grid_map/window_size_z", 8.0);
  // Upstream's whole-mission box: still declared so launch files written
  // for upstream load, and ignored, because memory follows the window alone.
  node_->declare_parameter("grid_map/map_size_x", -1.0);
  node_->declare_parameter("grid_map/map_size_y", -1.0);
  node_->declare_parameter("grid_map/map_size_z", -1.0);
  node_->declare_parameter("grid_map/obstacles_inflation", 0.1);
  node_->declare_parameter("grid_map/p_hit", 0.70);
  node_->declare_parameter("grid_map/p_miss", 0.35);
  node_->declare_parameter("grid_map/p_min", 0.12);
  node_->declare_parameter("grid_map/p_max", 0.97);
  node_->declare_parameter("grid_map/p_occ", 0.80);
  node_->declare_parameter("grid_map/max_ray_length", 8.0);
  node_->declare_parameter("grid_map/virtual_ceil_height", -1.0);
  node_->declare_parameter("grid_map/flight_band_enabled", false);
  node_->declare_parameter("grid_map/flight_band_min", 0.8);
  node_->declare_parameter("grid_map/flight_band_max", 2.5);
  node_->declare_parameter("grid_map/base_height", 0.12);
  node_->declare_parameter("grid_map/visualization_truncate_height", 100.0);
  node_->declare_parameter("grid_map/frame_id", std::string("world"));
  node_->declare_parameter("grid_map/odom_depth_timeout", 1.0);

  node_->get_parameter("grid_map/resolution", mp_.resolution_);
  node_->get_parameter("grid_map/window_size_x", mp_.window_size_(0));
  node_->get_parameter("grid_map/window_size_y", mp_.window_size_(1));
  node_->get_parameter("grid_map/window_size_z", mp_.window_size_(2));
  node_->get_parameter("grid_map/obstacles_inflation", mp_.obstacles_inflation_);
  node_->get_parameter("grid_map/p_hit", mp_.p_hit_);
  node_->get_parameter("grid_map/p_miss", mp_.p_miss_);
  node_->get_parameter("grid_map/p_min", mp_.p_min_);
  node_->get_parameter("grid_map/p_max", mp_.p_max_);
  node_->get_parameter("grid_map/p_occ", mp_.p_occ_);
  node_->get_parameter("grid_map/max_ray_length", mp_.max_ray_length_);
  node_->get_parameter("grid_map/virtual_ceil_height", mp_.virtual_ceil_height_);
  node_->get_parameter("grid_map/flight_band_enabled", mp_.flight_band_enabled_);
  node_->get_parameter("grid_map/flight_band_min", mp_.flight_band_min_);
  node_->get_parameter("grid_map/flight_band_max", mp_.flight_band_max_);
  node_->get_parameter("grid_map/base_height", mp_.base_height_);
  if (mp_.flight_band_enabled_ &&
      (!std::isfinite(mp_.flight_band_min_) || !std::isfinite(mp_.flight_band_max_) ||
       !std::isfinite(mp_.base_height_) || mp_.flight_band_min_ < 0.0 ||
       mp_.flight_band_max_ <= mp_.flight_band_min_ || mp_.base_height_ < 0.0))
    throw std::invalid_argument("invalid ground-relative flight band");
  node_->get_parameter("grid_map/visualization_truncate_height", mp_.visualization_truncate_height_);
  node_->get_parameter("grid_map/frame_id", mp_.frame_id_);
  node_->get_parameter("grid_map/odom_depth_timeout", mp_.sensor_timeout_);

  mp_.prob_hit_log_ = logOdds(mp_.p_hit_);
  mp_.prob_miss_log_ = logOdds(mp_.p_miss_);
  mp_.clamp_min_log_ = logOdds(mp_.p_min_);
  mp_.clamp_max_log_ = logOdds(mp_.p_max_);
  mp_.min_occupancy_log_ = logOdds(mp_.p_occ_);
  mp_.unknown_flag_ = 0.01f;

  Eigen::Vector3i voxels;
  for (int i = 0; i < 3; ++i)
    voxels(i) = std::max(1, int(std::ceil(mp_.window_size_(i) / mp_.resolution_ - 1e-9)));
  window_ = RollingWindow(mp_.resolution_, voxels);

  if (mp_.flight_band_enabled_)
  {
    column_ground_.resize(voxels.x() * voxels.y());
    column_ground_generation_.assign(column_ground_.size(), 0);
    flight_band_pub_ = node_->create_publisher<std_msgs::msg::Float64MultiArray>(
        "grid_map/flight_band", 10);
  }
  const std::size_t cells = window_.cells();
  md_.occupancy_buffer_.assign(cells, mp_.clamp_min_log_ - mp_.unknown_flag_);
  md_.occupancy_buffer_inflate_.assign(cells, 0);
  md_.count_hit_.assign(cells, 0);
  md_.count_hit_and_miss_.assign(cells, 0);
  md_.flag_traverse_.assign(cells, -1);
  md_.flag_rayend_.assign(cells, -1);
  md_.raycast_num_ = 0;
  md_.has_odom_ = false;
  md_.has_cloud_ = false;
  md_.local_updated_ = false;
  md_.flag_sensor_timeout_ = false;
  md_.sensor_pos_.setZero();
  window_.recenter(Eigen::Vector3d::Zero());
  md_.local_bound_min_ = window_.minIndex();
  md_.local_bound_max_ = window_.minIndex();

  callback_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  rclcpp::SubscriptionOptions sensing_options;
  sensing_options.callback_group = callback_group_;
  odom_sub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
      "grid_map/odom", 10, std::bind(&GridMap::odomCallback, this, std::placeholders::_1), sensing_options);
  cloud_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(
      node_, "grid_map/cloud", rclcpp::QoS(10).get_rmw_qos_profile(), sensing_options);
  origin_sub_ = std::make_shared<message_filters::Subscriber<geometry_msgs::msg::PointStamped>>(
      node_, "grid_map/cloud_origin", rclcpp::QoS(10).get_rmw_qos_profile(), sensing_options);
  sync_cloud_origin_ = std::make_shared<message_filters::Synchronizer<SyncPolicyCloudOrigin>>(
      SyncPolicyCloudOrigin(20), *cloud_sub_, *origin_sub_);
  sync_cloud_origin_->registerCallback(
      std::bind(&GridMap::cloudOriginCallback, this, std::placeholders::_1, std::placeholders::_2));

  timeout_timer_ = node_->create_wall_timer(std::chrono::milliseconds(50),
                                            std::bind(&GridMap::checkSensorTimeout, this), callback_group_);
  vis_timer_ = node_->create_wall_timer(std::chrono::milliseconds(110), [this]() {
    auto guard = lock();
    publishMapInflate(true);
    publishMap();
    double low, high;
    if (flightBandLimits(drone_pos_, low, high))
    {
      std_msgs::msg::Float64MultiArray band;
      band.data = {low, high};
      flight_band_pub_->publish(band);
    }
  }, callback_group_);

  map_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("grid_map/occupancy", 10);
  map_inf_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("grid_map/occupancy_inflate", 10);

  RCLCPP_INFO(node_->get_logger(), "rolling grid map: %d x %d x %d voxels of %.2f m (%zu cells)",
              voxels(0), voxels(1), voxels(2), mp_.resolution_, cells);
}

void GridMap::recenter(const Eigen::Vector3d &pos)
{
  auto guard = lock();
  const Eigen::Vector3i old_min = window_.minIndex();
  const Eigen::Vector3i old_max = window_.maxIndex();
  const auto entered = window_.recenter(pos);
  for (const RollingWindow::Box &box : entered)
    resetBox(box.first, box.second);

  const int step = int(std::ceil(mp_.obstacles_inflation_ / mp_.resolution_ - 1e-9));
  const Eigen::Vector3i margin = Eigen::Vector3i::Constant(step);
  for (const RollingWindow::Box &box : entered)
    inflateBox(window_.clamp(box.first - margin), window_.clamp(box.second + margin));
  // Departed occupancy can have inflated retained cells at the opposite edge.
  // Refresh that band too, using only occupancy still inside the new window.
  if (step > 0 && !entered.empty())
    for (int axis = 0; axis < 3; ++axis)
    {
      Eigen::Vector3i lo = window_.minIndex();
      Eigen::Vector3i hi = window_.maxIndex();
      if (lo(axis) > old_min(axis))
        hi(axis) = std::min(hi(axis), lo(axis) + step - 1);
      else if (hi(axis) < old_max(axis))
        lo(axis) = std::max(lo(axis), hi(axis) - step + 1);
      else
        continue;
      inflateBox(lo, hi);
    }
  md_.local_bound_min_ = window_.clamp(md_.local_bound_min_);
  md_.local_bound_max_ = window_.clamp(md_.local_bound_max_);
  if (mp_.flight_band_enabled_)
  {
    drone_pos_ = pos;
    if (!have_ground_reference_)
    {
      // Deployment starts on known ground. Subsequent levels come from lidar,
      // never from an airborne restart's guessed absolute world height.
      last_ground_ = pos.z() - mp_.base_height_;
      have_ground_reference_ = true;
    }
    ++ground_generation_;
    updateGroundReference();
  }
}

void GridMap::resetBox(const Eigen::Vector3i &min_id, const Eigen::Vector3i &max_id)
{
  for (int x = min_id(0); x <= max_id(0); ++x)
    for (int y = min_id(1); y <= max_id(1); ++y)
      for (int z = min_id(2); z <= max_id(2); ++z)
      {
        const std::size_t address = window_.address(Eigen::Vector3i(x, y, z));
        md_.occupancy_buffer_[address] = mp_.clamp_min_log_ - mp_.unknown_flag_;
        md_.occupancy_buffer_inflate_[address] = 0;
        md_.count_hit_[address] = 0;
        md_.count_hit_and_miss_[address] = 0;
        md_.flag_traverse_[address] = -1;
        md_.flag_rayend_[address] = -1;
      }
}

void GridMap::odomCallback(const nav_msgs::msg::Odometry::SharedPtr odom)
{
  auto guard = lock();
  const auto &p = odom->pose.pose.position;
  if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
    return;
  if (!md_.has_odom_ && !md_.has_cloud_)
    md_.last_cloud_ = std::chrono::steady_clock::now();
  md_.has_odom_ = true;
  recenter(Eigen::Vector3d(p.x, p.y, p.z));
}

void GridMap::cloudOriginCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &cloud,
                                  const geometry_msgs::msg::PointStamped::ConstSharedPtr &origin)
{
  pcl::PointCloud<pcl::PointXYZ> latest;
  pcl::fromROSMsg(*cloud, latest);
  vector<Eigen::Vector3d> points;
  points.reserve(latest.points.size());
  for (const pcl::PointXYZ &pt : latest.points)
    points.emplace_back(pt.x, pt.y, pt.z);
  inputCloud(points, Eigen::Vector3d(origin->point.x, origin->point.y, origin->point.z));
}

void GridMap::inputCloud(const vector<Eigen::Vector3d> &points, const Eigen::Vector3d &sensor_origin)
{
  auto guard = lock();
  md_.has_cloud_ = true;
  md_.last_cloud_ = std::chrono::steady_clock::now();
  md_.flag_sensor_timeout_ = false;
  if (!sensor_origin.allFinite() || !window_.contains(sensor_origin))
    return;
  md_.sensor_pos_ = sensor_origin;
  md_.points_.clear();
  for (const Eigen::Vector3d &point : points)
    if (point.allFinite())
      md_.points_.push_back(point);
  if (md_.points_.empty())
    return;
  raycastProcess();
  if (md_.local_updated_)
    clearAndInflateLocalMap();
  md_.local_updated_ = false;
  if (mp_.flight_band_enabled_)
  {
    ++ground_generation_;
    updateGroundReference();
  }
}

double GridMap::columnGround(const Eigen::Vector3d &pos)
{
  Eigen::Vector3i id = window_.indexOf(pos);
  id.z() = window_.minIndex().z();
  if (!window_.contains(id))
    return std::numeric_limits<double>::quiet_NaN();
  const std::size_t column = window_.address(id) / window_.size().z();
  if (column_ground_generation_[column] == ground_generation_)
    return column_ground_[column];
  double ground = std::numeric_limits<double>::quiet_NaN();
  const double underside = drone_pos_.z() - mp_.base_height_;
  const double bottom = underside - mp_.flight_band_max_ - 0.5;
  for (int z = window_.maxIndex().z(); z >= window_.minIndex().z(); --z)
  {
    id.z() = z;
    const double height = window_.centerOf(id).z() + mp_.resolution_ / 2.0;
    if (height >= underside) continue;
    if (height < bottom) break;
    if (md_.occupancy_buffer_[window_.address(id)] > mp_.min_occupancy_log_)
    {
      ground = height;
      break; // nearest observed surface below, not a lower storey's floor
    }
  }
  column_ground_generation_[column] = ground_generation_;
  return column_ground_[column] = ground;
}

void GridMap::updateGroundReference()
{
  const double ground = columnGround(drone_pos_);
  if (std::isfinite(ground)) last_ground_ = ground;
}

bool GridMap::flightBandLimits(const Eigen::Vector3d &pos, double &low, double &high)
{
  auto guard = lock();
  if (!mp_.flight_band_enabled_ || !have_ground_reference_) return false;
  const double observed = columnGround(pos);
  const double ground = std::isfinite(observed) ? observed : last_ground_;
  low = ground + mp_.flight_band_min_ + mp_.base_height_;
  high = ground + mp_.flight_band_max_ + mp_.base_height_;
  return true;
}

void GridMap::checkSensorTimeout()
{
  auto guard = lock();
  if (!md_.has_cloud_ && !md_.has_odom_)
    return;
  const double gap =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - md_.last_cloud_).count();
  md_.flag_sensor_timeout_ = gap > mp_.sensor_timeout_;
  if (md_.flag_sensor_timeout_)
    RCLCPP_ERROR_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                          "no lidar cloud for %.1f s", gap);
}

int GridMap::setCacheOccupancy(const Eigen::Vector3d &pos, int occ)
{
  const Eigen::Vector3i id = window_.indexOf(pos);
  if (!window_.contains(id))
    return INVALID_IDX;
  const int address = int(window_.address(id));
  md_.count_hit_and_miss_[address] += 1;
  if (md_.count_hit_and_miss_[address] == 1)
    md_.cache_voxel_.push(id);
  if (occ == 1)
    md_.count_hit_[address] += 1;
  return address;
}

void GridMap::raycastProcess()
{
  md_.raycast_num_ += 1;
  Eigen::Vector3d low = md_.sensor_pos_;
  Eigen::Vector3d high = md_.sensor_pos_;
  RayCaster raycaster;
  const Eigen::Vector3d half(0.5, 0.5, 0.5);
  Eigen::Vector3d ray_pt;

  for (Eigen::Vector3d pt_w : md_.points_)
  {
    const Eigen::Vector3d ray = pt_w - md_.sensor_pos_;
    const double length = ray.norm();
    const bool hit = window_.contains(pt_w) && length <= mp_.max_ray_length_;
    if (!hit)
    {
      // Returns beyond the window still observe free space. Clip the miss
      // to the nearer of the ray limit and window edge, never adding a hit.
      double fraction = length > 0.0 ? std::min(1.0, mp_.max_ray_length_ / length) : 0.0;
      for (int axis = 0; axis < 3; ++axis)
      {
        if (ray(axis) > 0.0)
          fraction = std::min(fraction, (window_.maxBound()(axis) - md_.sensor_pos_(axis)) / ray(axis));
        else if (ray(axis) < 0.0)
          fraction = std::min(fraction, (window_.minBound()(axis) - md_.sensor_pos_(axis)) / ray(axis));
      }
      // The maximum bound is exclusive; stay just inside for voxel indexing.
      pt_w = md_.sensor_pos_ + ray * std::max(0.0, fraction - 1e-8);
    }
    int vox_idx = setCacheOccupancy(pt_w, hit ? 1 : 0);
    low = low.cwiseMin(pt_w);
    high = high.cwiseMax(pt_w);

    if (vox_idx != INVALID_IDX)
    {
      if (md_.flag_rayend_[vox_idx] == md_.raycast_num_)
        continue;
      md_.flag_rayend_[vox_idx] = md_.raycast_num_;
    }

    raycaster.setInput(pt_w / mp_.resolution_, md_.sensor_pos_ / mp_.resolution_);
    while (raycaster.step(ray_pt))
    {
      const Eigen::Vector3d tmp = (ray_pt + half) * mp_.resolution_;
      vox_idx = setCacheOccupancy(tmp, 0);
      if (vox_idx != INVALID_IDX)
      {
        if (md_.flag_traverse_[vox_idx] == md_.raycast_num_)
          break;
        md_.flag_traverse_[vox_idx] = md_.raycast_num_;
      }
    }
  }

  md_.local_bound_min_ = window_.clamp(window_.indexOf(low));
  md_.local_bound_max_ = window_.clamp(window_.indexOf(high));
  md_.local_updated_ = true;

  while (!md_.cache_voxel_.empty())
  {
    const Eigen::Vector3i idx = md_.cache_voxel_.front();
    md_.cache_voxel_.pop();
    const std::size_t address = window_.address(idx);
    const float update =
        md_.count_hit_[address] >= md_.count_hit_and_miss_[address] - md_.count_hit_[address]
            ? mp_.prob_hit_log_
            : mp_.prob_miss_log_;
    md_.count_hit_[address] = md_.count_hit_and_miss_[address] = 0;

    float &value = md_.occupancy_buffer_[address];
    if (update >= 0 && value >= mp_.clamp_max_log_)
      continue;
    if (update <= 0 && value <= mp_.clamp_min_log_)
    {
      value = mp_.clamp_min_log_;
      continue;
    }
    value = std::min(std::max(value + update, mp_.clamp_min_log_), mp_.clamp_max_log_);
  }
}

void GridMap::clearAndInflateLocalMap()
{
  const int step = int(std::ceil(mp_.obstacles_inflation_ / mp_.resolution_ - 1e-9));
  const Eigen::Vector3i margin = Eigen::Vector3i::Constant(step);
  // Inflation is recomputed where it can have changed: the updated box plus
  // the inflation radius; occupied voxels up to one more radius out feed it.
  const Eigen::Vector3i clear_min = window_.clamp(md_.local_bound_min_ - margin);
  const Eigen::Vector3i clear_max = window_.clamp(md_.local_bound_max_ + margin);
  inflateBox(clear_min, clear_max);
}

void GridMap::inflateBox(const Eigen::Vector3i &clear_min, const Eigen::Vector3i &clear_max)
{
  const int step = int(std::ceil(mp_.obstacles_inflation_ / mp_.resolution_ - 1e-9));
  const Eigen::Vector3i margin = Eigen::Vector3i::Constant(step);
  const Eigen::Vector3i scan_min = window_.clamp(clear_min - margin);
  const Eigen::Vector3i scan_max = window_.clamp(clear_max + margin);

  for (int x = clear_min(0); x <= clear_max(0); ++x)
    for (int y = clear_min(1); y <= clear_max(1); ++y)
      for (int z = clear_min(2); z <= clear_max(2); ++z)
        md_.occupancy_buffer_inflate_[window_.address(Eigen::Vector3i(x, y, z))] = 0;

  for (int x = scan_min(0); x <= scan_max(0); ++x)
    for (int y = scan_min(1); y <= scan_max(1); ++y)
      for (int z = scan_min(2); z <= scan_max(2); ++z)
      {
        if (md_.occupancy_buffer_[window_.address(Eigen::Vector3i(x, y, z))] <= mp_.min_occupancy_log_)
          continue;
        // Iterate only the clear-box intersection. In particular, a dense
        // floor near a z boundary must not test all (2r+1)^3 neighbours.
        const Eigen::Vector3i voxel(x, y, z);
        const Eigen::Vector3i lo = clear_min.cwiseMax(voxel - margin);
        const Eigen::Vector3i hi = clear_max.cwiseMin(voxel + margin);
        const int z_size = window_.size()(2);
        const int count = hi(2) - lo(2) + 1;
        if (count <= 0)
          continue;  // an empty clear box: nothing to write
        for (int nx = lo(0); nx <= hi(0); ++nx)
          for (int ny = lo(1); ny <= hi(1); ++ny)
          {
            // z is contiguous except at its modulo seam: at most two fills.
            const std::size_t address = window_.address(Eigen::Vector3i(nx, ny, lo(2)));
            const int offset = int(address % z_size);
            const int first = std::min(count, z_size - offset);
            auto begin = md_.occupancy_buffer_inflate_.begin();
            std::fill_n(begin + address, first, 1);
            if (first < count)
              std::fill_n(begin + (address - offset), count - first, 1);
          }
      }

  // Optional ceiling to limit flight height (odometry-frame z; off below -0.5).
  if (mp_.virtual_ceil_height_ > -0.5)
  {
    const int ceil_z = int(std::floor(mp_.virtual_ceil_height_ / mp_.resolution_)) - 1;
    if (ceil_z >= clear_min(2) && ceil_z <= clear_max(2))
      for (int x = clear_min(0); x <= clear_max(0); ++x)
        for (int y = clear_min(1); y <= clear_max(1); ++y)
          md_.occupancy_buffer_inflate_[window_.address(Eigen::Vector3i(x, y, ceil_z))] = 1;
  }
}

void GridMap::publishMap()
{
  auto guard = lock();
  if (map_pub_->get_subscription_count() <= 0)
    return;
  pcl::PointCloud<pcl::PointXYZ> cloud;
  for (int x = md_.local_bound_min_(0); x <= md_.local_bound_max_(0); ++x)
    for (int y = md_.local_bound_min_(1); y <= md_.local_bound_max_(1); ++y)
      for (int z = md_.local_bound_min_(2); z <= md_.local_bound_max_(2); ++z)
      {
        const Eigen::Vector3i id(x, y, z);
        if (md_.occupancy_buffer_[window_.address(id)] < mp_.min_occupancy_log_)
          continue;
        const Eigen::Vector3d pos = window_.centerOf(id);
        if (pos(2) > mp_.visualization_truncate_height_)
          continue;
        cloud.push_back(pcl::PointXYZ(pos(0), pos(1), pos(2)));
      }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.frame_id_;
  sensor_msgs::msg::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  map_pub_->publish(cloud_msg);
}

void GridMap::publishMapInflate(bool all_info)
{
  auto guard = lock();
  if (map_inf_pub_->get_subscription_count() <= 0)
    return;
  const Eigen::Vector3i low = all_info ? window_.minIndex() : md_.local_bound_min_;
  const Eigen::Vector3i high = all_info ? window_.maxIndex() : md_.local_bound_max_;
  pcl::PointCloud<pcl::PointXYZ> cloud;
  for (int x = low(0); x <= high(0); ++x)
    for (int y = low(1); y <= high(1); ++y)
      for (int z = low(2); z <= high(2); ++z)
      {
        const Eigen::Vector3i id(x, y, z);
        if (md_.occupancy_buffer_inflate_[window_.address(id)] == 0)
          continue;
        const Eigen::Vector3d pos = window_.centerOf(id);
        if (pos(2) > mp_.visualization_truncate_height_)
          continue;
        cloud.push_back(pcl::PointXYZ(pos(0), pos(1), pos(2)));
      }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.frame_id_;
  sensor_msgs::msg::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  map_inf_pub_->publish(cloud_msg);
}


bool GridMap::escapeSegmentSafe(const Eigen::Vector3d &start, const Eigen::Vector3d &end)
{
  auto guard = lock();
  const double bound = mp_.obstacles_inflation_ + 2 * mp_.resolution_;
  const Eigen::Vector3d delta = end - start;
  if (!start.allFinite() || !end.allFinite() || delta.norm() > bound + 1e-9 ||
      delta.norm() < 1e-9 || getOccupancy(start) != 0 || getInflateOccupancy(end) != 0)
    return false;
  std::vector<Eigen::Vector3d> obstacles;
  const Eigen::Vector3i lo = window_.indexOf(start.cwiseMin(end) - Eigen::Vector3d::Constant(bound));
  const Eigen::Vector3i hi = window_.indexOf(start.cwiseMax(end) + Eigen::Vector3d::Constant(bound));
  const double half = mp_.resolution_ * .5;
  for (int x=lo.x(); x<=hi.x(); ++x)
    for (int y=lo.y(); y<=hi.y(); ++y)
      for (int z=lo.z(); z<=hi.z(); ++z) {
        const Eigen::Vector3i id(x,y,z);
        if (!window_.contains(id)) continue;
        if (md_.occupancy_buffer_[window_.address(id)] <= mp_.min_occupancy_log_) continue;
        const Eigen::Vector3d c = window_.centerOf(id);
        obstacles.push_back(c);
        // Exact segment/raw-voxel intersection, including touching.
        double first=0, last=1;
        for (int axis=0; axis<3; ++axis) {
          if (std::abs(delta[axis]) < 1e-12) {
            if (std::abs(start[axis]-c[axis]) > half) { first=2; break; }
          } else {
            double a=(c[axis]-half-start[axis])/delta[axis];
            double b=(c[axis]+half-start[axis])/delta[axis];
            if (a>b) std::swap(a,b);
            first=std::max(first,a); last=std::min(last,b);
          }
        }
        if (first<=last) return false;
        // Strictly outward from every raw cube inside the initial inflation.
        const Eigen::Vector3d closest = start.cwiseMax(c-Eigen::Vector3d::Constant(half))
                                           .cwiseMin(c+Eigen::Vector3d::Constant(half));
        const Eigen::Vector3d away = start-closest;
        if (away.cwiseAbs().maxCoeff() <= mp_.obstacles_inflation_ + 1e-9 &&
            away.dot(delta) <= 1e-10) return false;
      }
  if (obstacles.empty()) return false;
  double previous = -1;
  bool left = false;
  const int count = std::max(1, int(std::ceil(delta.norm() / (mp_.resolution_ * .1))));
  for (int i=0; i<=count; ++i) {
    const Eigen::Vector3d p = start + delta * (double(i)/count);
    if (!isInMap(p) || getOccupancy(p) != 0) return false;
    double low, high;
    if (flightBandLimits(p, low, high) && (p.z()<low || p.z()>high)) return false;
    double clearance = std::numeric_limits<double>::infinity();
    for (const auto &c : obstacles)
      clearance = std::min(clearance, ((p-c).cwiseAbs()-Eigen::Vector3d::Constant(half)).cwiseMax(0).norm());
    if (!left && i>0 && clearance <= previous + 1e-10) return false;
    const bool occupied = getInflateOccupancy(p) != 0;
    if (left && occupied) return false;
    if (!occupied) left = true;
    previous = clearance;
  }
  return left;
}

bool GridMap::inflatedEscape(const Eigen::Vector3d &start, const Eigen::Vector3d &preferred,
                            Eigen::Vector3d &end)
{
  auto guard = lock();
  if (getOccupancy(start) != 0 || getInflateOccupancy(start) != 1) return false;
  std::vector<Eigen::Vector3d> directions;
  if ((preferred-start).norm()>1e-9) directions.push_back((preferred-start).normalized());
  for (int x=-1; x<=1; ++x)
    for (int y=-1; y<=1; ++y)
      for (int z=-1; z<=1; ++z)
        if (x || y || z) directions.push_back(Eigen::Vector3d(x,y,z).normalized());
  const double bound = mp_.obstacles_inflation_ + 2 * mp_.resolution_;
  for (const auto &direction : directions) {
    for (double distance=mp_.resolution_*.25; distance<=bound+1e-9; distance+=mp_.resolution_*.25) {
      const Eigen::Vector3d candidate = start + distance*direction;
      if (!escapeSegmentSafe(start, candidate)) continue;
      // Distance to the inflated voxel UNION, not distance travelled along
      // an oblique ray. A full cell of normal clearance tolerates tracking
      // error in any direction, including towards a corner of the boundary.
      const Eigen::Vector3i center = window_.indexOf(candidate);
      bool margin = true;
      for (int x=-2; x<=2 && margin; ++x)
        for (int y=-2; y<=2 && margin; ++y)
          for (int z=-2; z<=2 && margin; ++z) {
            const Eigen::Vector3i id = center + Eigen::Vector3i(x,y,z);
            if (!window_.contains(id)) { margin=false; break; }
            if (md_.occupancy_buffer_inflate_[window_.address(id)] == 0) continue;
            const double clearance = ((candidate-window_.centerOf(id)).cwiseAbs()
                - Eigen::Vector3d::Constant(mp_.resolution_*.5)).cwiseMax(0).norm();
            if (clearance < mp_.resolution_ - 1e-9) margin=false;
          }
      if (margin) {
        end=candidate;
        return true;
      }
    }
  }
  return false;
}
