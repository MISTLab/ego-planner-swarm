// The rolling grid map (SwarmDeck drone scout design §3.3): occupancy stays
// correct as the window moves, the edges are blocked, and memory does not
// depend on the mission size.

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <thread>

#include <rclcpp/rclcpp.hpp>

#include "plan_env/grid_map.h"
#include "plan_env/flight_band_check.h"

namespace {

class RollingGridMap : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
  static void TearDownTestSuite() { rclcpp::shutdown(); }

  GridMap::Ptr make(double map_size = 50.0, double inflation = 0.0, double timeout = 1.0) {
    rclcpp::NodeOptions options;
    options.parameter_overrides({
        {"grid_map/resolution", 0.1},
        {"grid_map/odom_depth_timeout", timeout},
        {"grid_map/window_size_x", 4.0},
        {"grid_map/window_size_y", 4.0},
        {"grid_map/window_size_z", 2.0},
        {"grid_map/map_size_x", map_size},
        {"grid_map/map_size_y", map_size},
        {"grid_map/map_size_z", map_size},
        {"grid_map/obstacles_inflation", inflation},
        {"grid_map/max_ray_length", 3.0},
        {"grid_map/p_hit", 0.9},
        {"grid_map/p_miss", 0.3},
        {"grid_map/virtual_ceil_height", -1.0},
    });
    node_ = std::make_shared<rclcpp::Node>(
        "grid_map_test_" + std::to_string(next_node_++), options);
    auto map = std::make_shared<GridMap>();
    map->initMap(node_);
    return map;
  }

  static void observe(GridMap& map, const Eigen::Vector3d& point,
                      const Eigen::Vector3d& origin = Eigen::Vector3d::Zero()) {
    for (int i = 0; i < 3; ++i) map.inputCloud({point}, origin);
  }

  void checkInflationAfterShift(const Eigen::Vector3d &shift) {
    auto map = make(50.0, 0.2);
    // z = 0.05 (index 0): its inflation spans z -2..2, across the z modulo
    // seam between -1 and 0, so both fill spans are checked.
    const std::vector<Eigen::Vector3d> obstacles = {
        {1.85, 1.85, 0.85}, {-1.85, -1.85, -0.85},
        {0.05, 0.05, 0.85}, {0.05, 0.05, -0.85}, {1.05, -1.05, 0.05}};
    for (const auto &point : obstacles) observe(*map, point);
    for (const auto &point : obstacles) ASSERT_EQ(map->getOccupancy(point), 1);
    map->recenter(shift);
    const Eigen::Vector3d low = map->windowMin();
    const Eigen::Vector3d high = map->windowMax();
    for (int x = 0; x < 40; ++x)
      for (int y = 0; y < 40; ++y)
        for (int z = 0; z < 20; ++z) {
          const Eigen::Vector3d point = low + Eigen::Vector3d(x + 0.5, y + 0.5, z + 0.5) * 0.1;
          bool expected = false;
          for (const auto &obstacle : obstacles)
            if ((obstacle.array() >= low.array()).all() && (obstacle.array() < high.array()).all())
              expected |= (obstacle - point).cwiseAbs().maxCoeff() < 0.200001;
          ASSERT_EQ(map->getInflateOccupancy(point), expected ? 1 : 0)
              << "point " << point.transpose() << ", shift " << shift.transpose();
        }
  }

  rclcpp::Node::SharedPtr node_;
  static int next_node_;
};

int RollingGridMap::next_node_ = 0;

TEST_F(RollingGridMap, LogOddsAreStoredAsFloat) {
  static_assert(std::is_same<GridMap::OccupancyValue, float>::value,
                "log-odds must be float");
  SUCCEED();
}

TEST_F(RollingGridMap, MemoryFollowsTheWindowNotTheMission) {
  EXPECT_EQ(make(50.0)->bufferCells(), 40u * 40u * 20u);
  EXPECT_EQ(make(5000.0)->bufferCells(), 40u * 40u * 20u);
}

TEST_F(RollingGridMap, AnObstacleStaysWhileInsideAndIsForgottenOnceItLeaves) {
  auto map = make();
  map->recenter(Eigen::Vector3d::Zero());  // x in [-2, 2)
  const Eigen::Vector3d wall(1.05, 0.05, 0.05);
  observe(*map, wall);
  EXPECT_EQ(map->getInflateOccupancy(wall), 1);
  map->recenter(Eigen::Vector3d(1.5, 0.0, 0.0));  // x in [-0.5, 3.5)
  EXPECT_EQ(map->getInflateOccupancy(wall), 1);
  map->recenter(Eigen::Vector3d(3.5, 0.0, 0.0));  // x in [1.5, 5.5)
  EXPECT_EQ(map->getInflateOccupancy(wall), -1);
  map->recenter(Eigen::Vector3d::Zero());  // it re-enters, unknown
  EXPECT_EQ(map->getInflateOccupancy(wall), 0);
  EXPECT_TRUE(map->isUnknown(wall));
}

TEST_F(RollingGridMap, NoAliasingAcrossTheWrap) {
  auto map = make();
  map->recenter(Eigen::Vector3d::Zero());
  const Eigen::Vector3d behind(-1.05, 0.05, 0.05);  // index -11
  observe(*map, behind);
  map->recenter(Eigen::Vector3d(1.5, 0.0, 0.0));
  // Index 29 is stored where -11 was (40 voxels apart).
  const Eigen::Vector3d ahead(2.95, 0.05, 0.05);
  EXPECT_EQ(map->getInflateOccupancy(ahead), 0);
  observe(*map, ahead, Eigen::Vector3d(1.5, 0.0, 0.0));
  EXPECT_EQ(map->getInflateOccupancy(ahead), 1);
  EXPECT_EQ(map->getInflateOccupancy(behind), -1);
}

TEST_F(RollingGridMap, OutsideTheWindowIsBlockedAndInsideUnknownIsFree) {
  auto map = make();
  map->recenter(Eigen::Vector3d::Zero());
  // Planners treat any non-zero value as occupied: a path across the edge
  // stops at the edge instead of running into unmapped space.
  EXPECT_EQ(map->getInflateOccupancy(Eigen::Vector3d(2.5, 0.0, 0.0)), -1);
  EXPECT_EQ(map->getInflateOccupancy(Eigen::Vector3d(0.0, 0.0, 1.5)), -1);
  EXPECT_EQ(map->getInflateOccupancy(Eigen::Vector3d(1.5, 0.0, 0.0)), 0);
}

TEST_F(RollingGridMap, OnlyHitsInsideTheWindowAreFedIn) {
  auto map = make();
  map->recenter(Eigen::Vector3d::Zero());
  observe(*map, Eigen::Vector3d(10.0, 0.05, 0.05));
  EXPECT_FALSE(map->isUnknown(Eigen::Vector3d(1.05, 0.05, 0.05)));
  EXPECT_FALSE(map->isUnknown(Eigen::Vector3d(1.95, 0.05, 0.05)));
  EXPECT_EQ(map->getOccupancy(Eigen::Vector3d(1.95, 0.05, 0.05)), 0);
  EXPECT_EQ(map->getOccupancy(Eigen::Vector3d(10.0, 0.05, 0.05)), -1);
  EXPECT_EQ(map->getInflateOccupancy(Eigen::Vector3d(1.95, 0.05, 0.05)), 0);
}

TEST_F(RollingGridMap, OutsideReturnsRespectRayLengthAndNegativeEdges) {
  auto map = make();
  observe(*map, Eigen::Vector3d(-10.0, 0.05, 0.05), Eigen::Vector3d(1.5, 0.05, 0.05));
  EXPECT_FALSE(map->isUnknown(Eigen::Vector3d(-1.45, 0.05, 0.05)));
  EXPECT_TRUE(map->isUnknown(Eigen::Vector3d(-1.75, 0.05, 0.05)));
  observe(*map, Eigen::Vector3d(0.05, 0.05, -10.0));
  EXPECT_FALSE(map->isUnknown(Eigen::Vector3d(0.05, 0.05, -0.95)));
  EXPECT_EQ(map->getOccupancy(Eigen::Vector3d(0.05, 0.05, -0.95)), 0);
}

TEST_F(RollingGridMap, FreeSpaceIsCarvedUpToTheReturn) {
  auto map = make();
  map->recenter(Eigen::Vector3d::Zero());
  observe(*map, Eigen::Vector3d(1.55, 0.05, 0.05));
  EXPECT_FALSE(map->isUnknown(Eigen::Vector3d(0.75, 0.05, 0.05)));
  EXPECT_EQ(map->getOccupancy(Eigen::Vector3d(0.75, 0.05, 0.05)), 0);
  EXPECT_EQ(map->getOccupancy(Eigen::Vector3d(1.55, 0.05, 0.05)), 1);
}

TEST_F(RollingGridMap, DepartedObstacleLeavesNoInflationInRetainedCells) {
  auto map = make(50.0, 0.2);
  const Eigen::Vector3d obstacle(-1.95, 0.05, 0.05);
  const Eigen::Vector3d neighbour(-1.75, 0.05, 0.05);
  observe(*map, obstacle);
  ASSERT_EQ(map->getInflateOccupancy(neighbour), 1);
  map->recenter(Eigen::Vector3d(0.2, 0.0, 0.0));
  EXPECT_EQ(map->getInflateOccupancy(obstacle), -1);
  EXPECT_EQ(map->getInflateOccupancy(neighbour), 0);
}

TEST_F(RollingGridMap, RetainedObstacleInflatesNewCellsWithoutAnotherScan) {
  auto map = make(50.0, 0.2);
  const Eigen::Vector3d obstacle(1.95, 0.05, 0.05);
  const Eigen::Vector3d neighbour(2.05, 0.05, 0.05);
  observe(*map, obstacle);
  ASSERT_EQ(map->getInflateOccupancy(obstacle), 1);
  ASSERT_EQ(map->getInflateOccupancy(neighbour), -1);
  map->recenter(Eigen::Vector3d(0.2, 0.0, 0.0));
  EXPECT_EQ(map->getInflateOccupancy(obstacle), 1);
  EXPECT_EQ(map->getInflateOccupancy(neighbour), 1);
}

TEST_F(RollingGridMap, SensorTimeoutStartsAtFirstOdometryWithoutAPairedCloud) {
  auto map = make(50.0, 0.0, 0.1);
  auto odom = node_->create_publisher<nav_msgs::msg::Odometry>("grid_map/odom", 10);
  auto cloud = node_->create_publisher<sensor_msgs::msg::PointCloud2>("grid_map/cloud", 10);
  const auto spin_for = [&](int milliseconds, bool publish) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(milliseconds);
    while (std::chrono::steady_clock::now() < end) {
      if (publish) {
        odom->publish(nav_msgs::msg::Odometry());
        sensor_msgs::msg::PointCloud2 msg;
        msg.header.stamp = node_->now();
        cloud->publish(msg);  // No matching origin: this is not a valid pair.
      }
      rclcpp::spin_some(node_);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  };
  spin_for(200, false);
  EXPECT_FALSE(map->getOdomDepthTimeout());
  spin_for(500, true);
  ASSERT_TRUE(map->odomValid());
  EXPECT_TRUE(map->getOdomDepthTimeout());
  map->inputCloud({}, Eigen::Vector3d::Zero());
  EXPECT_FALSE(map->getOdomDepthTimeout());
  spin_for(200, true);
  EXPECT_TRUE(map->getOdomDepthTimeout());
}

TEST_F(RollingGridMap, ZShiftsRefreshAllAndOnlyRetainedInflation) {
  checkInflationAfterShift(Eigen::Vector3d(0.0, 0.0, 0.2));
  checkInflationAfterShift(Eigen::Vector3d(0.0, 0.0, -0.2));
}

TEST_F(RollingGridMap, DiagonalShiftsRefreshAllAndOnlyRetainedInflation) {
  checkInflationAfterShift(Eigen::Vector3d(0.2, 0.2, 0.2));
  checkInflationAfterShift(Eigen::Vector3d(-0.2, 0.2, -0.2));
}

TEST_F(RollingGridMap, FullJumpForgetsAllInflation) {
  checkInflationAfterShift(Eigen::Vector3d(5.0, -5.0, 3.0));
}

void recenterTiming(const std::string &name, const Eigen::Vector3d &shift, int wall_axis) {
  rclcpp::NodeOptions options;
  // Long rays populate the entire dense boundary plane, not just a lidar disc.
  options.parameter_overrides({{"grid_map/obstacles_inflation", 0.25},
                               {"grid_map/max_ray_length", 40.0}, {"grid_map/p_hit", 0.9}});
  auto node = std::make_shared<rclcpp::Node>("grid_map_timing_" + name, options);
  GridMap map;
  map.initMap(node);
  std::vector<Eigen::Vector3d> wall;
  const int axis_a = wall_axis == 0 ? 1 : 0;
  const int axis_b = wall_axis == 0 ? 2 : 1;
  const Eigen::Vector3i size(240, 240, 80);
  const Eigen::Vector3d low(-12.0, -12.0, -4.0);
  for (int a = 0; a < size(axis_a); ++a)
    for (int b = 0; b < size(axis_b); ++b) {
      Eigen::Vector3d point = low + Eigen::Vector3d::Constant(0.05);
      point(wall_axis) += (size(wall_axis) - 1) * 0.1;
      point(axis_a) += a * 0.1;
      point(axis_b) += b * 0.1;
      wall.push_back(point);
    }
  for (int i = 0; i < 3; ++i) map.inputCloud(wall, Eigen::Vector3d::Zero());
  double total_ms = 0.0, max_ms = 0.0;
  for (int i = 0; i < 100; ++i) {
    const auto start = std::chrono::steady_clock::now();
    map.recenter(i % 2 ? Eigen::Vector3d::Zero().eval() : shift);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    total_ms += ms;
    max_ms = std::max(max_ms, ms);
  }
  std::cout << "Default window " << name << " recenter: mean " << total_ms / 100.0
            << " ms, max " << max_ms << " ms (100 shifts)" << std::endl;
  // Loose regression bound, not a real-time guarantee. Opt-in: the margin is
  // small on tuf, and a slower or loaded runner can exceed it with no
  // regression.
  if (std::getenv("EGO_ASSERT_RECENTER_TIMING") != nullptr)
    EXPECT_LT(total_ms / 100.0, 20.0);
  else
    std::cout << "(set EGO_ASSERT_RECENTER_TIMING to assert mean < 20 ms)" << std::endl;
  EXPECT_EQ(map.bufferCells(), 240u * 240u * 80u);
}

TEST_F(RollingGridMap, DefaultWindowRecenterTiming) {
  recenterTiming("x", Eigen::Vector3d(0.1, 0.0, 0.0), 0);
}

TEST_F(RollingGridMap, DefaultWindowZRecenterTiming) {
  recenterTiming("z", Eigen::Vector3d(0.0, 0.0, 0.1), 2);
}

TEST_F(RollingGridMap, DefaultWindowDiagonalRecenterTiming) {
  recenterTiming("diagonal", Eigen::Vector3d(0.1, 0.1, 0.1), 2);
}

}  // namespace

class GroundRelativeBand : public RollingGridMap {
 protected:
  GridMap::Ptr band() {
    rclcpp::NodeOptions options;
    options.parameter_overrides({
        {"grid_map/window_size_x", 8.0}, {"grid_map/window_size_y", 8.0},
        {"grid_map/window_size_z", 12.0}, {"grid_map/p_hit", 0.9},
        {"grid_map/max_ray_length", 10.0}, {"grid_map/flight_band_enabled", true},
        {"grid_map/flight_band_min", 0.8}, {"grid_map/flight_band_max", 2.5},
        {"grid_map/base_height", 0.12}});
    node_ = std::make_shared<rclcpp::Node>("band_test_" + std::to_string(next_node_++), options);
    auto map = std::make_shared<GridMap>(); map->initMap(node_);
    // The first odometry is on known ground (base height 0.12).
    map->recenter({0.05, 0.05, 0.12});
    map->recenter({0.05, 0.05, 1.32});
    return map;
  }
};

TEST_F(GroundRelativeBand, BlocksWholeHalfSpacesNotJustOneCeilingVoxel) {
  auto map = band();
  EXPECT_EQ(map->getInflateOccupancy({1.05, 0.05, 0.8}), 1);
  EXPECT_EQ(map->getInflateOccupancy({1.05, 0.05, 1.5}), 0);
  EXPECT_EQ(map->getInflateOccupancy({1.05, 0.05, 2.7}), 1);
  EXPECT_EQ(map->getInflateOccupancy({1.05, 0.05, 5.0}), 1);
}

TEST_F(GroundRelativeBand, FollowsObservedRampAndSeparateLevels) {
  auto map = band();
  map->recenter({0.05, 0.05, 2.32});
  observe(*map, {1.05, 0.05, 0.55}, {0.05, 0.05, 2.32});
  observe(*map, {2.05, 0.05, 1.05}, {0.05, 0.05, 2.32});
  EXPECT_EQ(map->getInflateOccupancy({1.05, 0.05, 1.3}), 1);
  EXPECT_EQ(map->getInflateOccupancy({1.05, 0.05, 2.8}), 0);
  EXPECT_EQ(map->getInflateOccupancy({2.05, 0.05, 1.8}), 1);
  EXPECT_EQ(map->getInflateOccupancy({2.05, 0.05, 3.4}), 0);
  EXPECT_EQ(map->getInflateOccupancy({2.05, 0.05, 3.8}), 1);
}

TEST_F(GroundRelativeBand, UsesDeckAboveLowerFloorAndIgnoresCeiling) {
  auto map = band();
  observe(*map, {1.05, 0.05, -1.05});
  observe(*map, {1.05, 0.05, 0.05});
  observe(*map, {1.05, 0.05, 3.05});
  EXPECT_EQ(map->getInflateOccupancy({1.05, 0.05, 2.0}), 0);
  EXPECT_EQ(map->getInflateOccupancy({1.05, 0.05, 0.85}), 1);
  EXPECT_EQ(map->getInflateOccupancy({1.05, 0.05, 2.8}), 1);
}

TEST_F(GroundRelativeBand, ShaftAndUnseenColumnsKeepLastKnownLevel) {
  auto map = band();
  map->recenter({0.05, 0.05, 2.32});
  observe(*map, {0.05, 0.05, 1.05}, {0.05, 0.05, 2.32});
  // Establish the current column, then move over a drop deeper than 3 m.
  map->recenter({0.05, 0.05, 2.32});
  observe(*map, {2.05, 0.05, -2.05}, {0.05, 0.05, 2.32});
  map->recenter({2.05, 0.05, 2.32});
  EXPECT_EQ(map->getInflateOccupancy({2.05, 0.05, 3.4}), 0);
  EXPECT_EQ(map->getInflateOccupancy({2.05, 0.05, 1.8}), 1);
  map->recenter({40.05, 0.05, 2.32});
  EXPECT_EQ(map->getInflateOccupancy({40.05, 0.05, 3.4}), 0);
  EXPECT_EQ(map->getInflateOccupancy({40.05, 0.05, 1.8}), 1);
}

TEST_F(GroundRelativeBand, GroundIsTheObservedVoxelsTopSurface) {
  auto map = band();
  observe(*map, {1.05, 0.05, 0.05});
  double low, high;
  ASSERT_TRUE(map->flightBandLimits({1.05, 0.05, 1.5}, low, high));
  EXPECT_NEAR(low, 0.1 + 0.8 + 0.12, 1e-9);
  EXPECT_NEAR(high, 0.1 + 2.5 + 0.12, 1e-9);
}

TEST_F(GroundRelativeBand, PoseToleranceNeverUnblocksRealOccupancy) {
  auto map = band();
  const Eigen::Vector3d drift(1.05, 0.05, 2.65); // upper band 2.62
  EXPECT_EQ(map->getInflateOccupancy(drift), 1);
  EXPECT_EQ(map->getInflateOccupancy(drift, 0.1), 0);
  EXPECT_EQ(map->getInflateOccupancy({1.05, 0.05, 2.73}, 0.1), 1);
  observe(*map, drift);
  EXPECT_EQ(map->getInflateOccupancy(drift, 0.1), 1);
}

TEST_F(GroundRelativeBand, AcceptsOnlyARecoveringInitialBandPrefix) {
  auto map = band();
  FlightBandTrajectoryCheck check(*map);
  EXPECT_TRUE(check.accept({1.05, 0.05, 2.65}, 0.0, false));
  EXPECT_TRUE(check.accept({1.05, 0.05, 2.64}, 0.1, false));
  EXPECT_TRUE(check.accept({1.05, 0.05, 2.61}, 0.3, false));
  EXPECT_TRUE(check.accept({1.05, 0.05, 2.5}, 1.0, true));
  // Once inside, even a 1 cm excursion is forbidden.
  EXPECT_FALSE(check.accept({1.05, 0.05, 2.63}, 1.1, false));
}

TEST_F(GroundRelativeBand, RejectsEscapingAndOverduePrefixesAndOutsideEndpoints) {
  auto map = band();
  FlightBandTrajectoryCheck escaping(*map), overdue(*map), endpoint(*map);
  ASSERT_TRUE(escaping.accept({1.05, 0.05, 2.65}, 0.0, false));
  EXPECT_FALSE(escaping.accept({1.05, 0.05, 2.66}, 0.1, false));
  ASSERT_TRUE(overdue.accept({1.05, 0.05, 2.65}, 0.0, false));
  EXPECT_FALSE(overdue.accept({1.05, 0.05, 2.64}, 0.51, false));
  ASSERT_TRUE(endpoint.accept({1.05, 0.05, 2.65}, 0.0, false));
  EXPECT_FALSE(endpoint.accept({1.05, 0.05, 2.64}, 0.2, true));
}

TEST_F(GroundRelativeBand, RejectsOccupiedVoxelsDuringTheRecoveryPrefix) {
  auto map = band();
  observe(*map, {1.05, 0.05, 2.65});
  FlightBandTrajectoryCheck check(*map);
  EXPECT_FALSE(check.accept({1.05, 0.05, 2.65}, 0.0, false));
}

TEST_F(GroundRelativeBand, LeavesInBandOccupancyToUpstreamCollisionChecks) {
  auto map = band();
  const Eigen::Vector3d occupied(1.05, 0.05, 1.65);
  observe(*map, occupied);
  ASSERT_EQ(map->getInflateOccupancy(occupied), 1);
  FlightBandTrajectoryCheck check(*map);
  EXPECT_TRUE(check.accept({0.05, 0.05, 1.65}, 0.0, false));
  EXPECT_TRUE(check.accept(occupied, 1.0, false));
  EXPECT_TRUE(check.accept(occupied, 2.0, true));
}

TEST_F(GroundRelativeBand, BandOnlyStartCanCorrectVerticallyThroughObservedFreeColumn) {
  auto map = band();
  const Eigen::Vector3d low(1.05, .05, .82), inside(1.05, .05, 1.12);
  // Clear the correction column; the raw endpoint is beyond the escape bound.
  observe(*map, {1.05, .05, 4.05}, {1.05, .05, .52});
  ASSERT_EQ(map->getOccupancy(low), 0);
  ASSERT_EQ(map->getInflateOccupancy(low), 1);
  EXPECT_TRUE(map->escapeSegmentSafe(low, inside));
  Eigen::Vector3d end;
  ASSERT_TRUE(map->inflatedEscape(low, {3., .05, 1.12}, end));
  EXPECT_NEAR(end.x(), low.x(), 1e-9);
  EXPECT_NEAR(end.y(), low.y(), 1e-9);
  EXPECT_GE(end.z(), .92);
  EXPECT_FALSE(map->escapeSegmentSafe(low, {1.15, .05, 1.12}));
  EXPECT_FALSE(map->escapeSegmentSafe(low, {1.05, .05, .72}));
}

TEST_F(GroundRelativeBand, BandCorrectionRejectsUnknownAndInflatedColumns) {
  auto map = band();
  const Eigen::Vector3d low(1.05, .05, .82), inside(1.05, .05, 1.12);
  EXPECT_FALSE(map->escapeSegmentSafe(low, inside));
  observe(*map, {1.05, .05, 4.05}, {1.05, .05, .52});
  observe(*map, {1.15, .05, 1.12}, low);
  EXPECT_FALSE(map->escapeSegmentSafe(low, inside));
}

TEST_F(GroundRelativeBand, UpperBandCorrectionIsAlsoBoundedAndVertical) {
  auto map = band();
  const Eigen::Vector3d high(1.05, .05, 2.72), inside(1.05, .05, 2.42);
  observe(*map, {1.05, .05, 4.05}, {1.05, .05, 1.32});
  EXPECT_TRUE(map->escapeSegmentSafe(high, inside));
  EXPECT_FALSE(map->escapeSegmentSafe({1.05, .05, 3.42}, inside));
}

TEST_F(GroundRelativeBand, RaisedSupportBandCorrectionDoesNotEraseTheSupport) {
  auto map = band();
  observe(*map, {1.05, .05, .25}, {1.05, .05, 1.32});
  observe(*map, {1.05, .05, 4.05}, {1.05, .05, .52});
  const Eigen::Vector3d start(1.05, .05, 1.12), end(1.05, .05, 1.42);
  double low, high;
  ASSERT_TRUE(map->flightBandLimits(start, low, high));
  EXPECT_NEAR(low, 1.22, 1e-9);
  EXPECT_EQ(map->getInflateOccupancy(start), 1);
  EXPECT_TRUE(map->escapeSegmentSafe(start, end));
  EXPECT_EQ(map->getOccupancy({1.05, .05, .25}), 1);
}

TEST_F(GroundRelativeBand, BandCorrectionAdmissionCertifiesTheWholeTrackingTube) {
  auto map = band();
  const Eigen::Vector3d low(1.05, .05, .82), inside(1.05, .05, 1.12);
  // Only the centre column observed: the tracking tube's edge columns are
  // unknown, so admission must refuse what tracking would stop at once.
  observe(*map, {1.05, .05, 4.05}, {1.05, .05, .52});
  EXPECT_FALSE(map->bandEscapeTrackingSafe(low, inside, low));
  EXPECT_FALSE(map->escapeSegmentSafe(low, inside));
  for (double x : {.95, 1.15}) observe(*map, {x, .05, 4.05}, {x, .05, .52});
  for (double y : {-.05, .15}) observe(*map, {1.05, y, 4.05}, {1.05, y, .52});
  EXPECT_TRUE(map->bandEscapeTrackingSafe(low, inside, low));
  EXPECT_TRUE(map->escapeSegmentSafe(low, inside));
}
