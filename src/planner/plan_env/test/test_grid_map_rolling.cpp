// The rolling grid map (SwarmDeck drone scout design §3.3): occupancy stays
// correct as the window moves, the edges are blocked, and memory does not
// depend on the mission size.

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <type_traits>

#include <rclcpp/rclcpp.hpp>

#include "plan_env/grid_map.h"

namespace {

class RollingGridMap : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
  static void TearDownTestSuite() { rclcpp::shutdown(); }

  GridMap::Ptr make(double map_size = 50.0, double inflation = 0.0) {
    rclcpp::NodeOptions options;
    options.parameter_overrides({
        {"grid_map/resolution", 0.1},
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

TEST_F(RollingGridMap, DefaultWindowRecenterTiming) {
  rclcpp::NodeOptions options;
  options.parameter_overrides({{"grid_map/obstacles_inflation", 0.25}});
  auto node = std::make_shared<rclcpp::Node>("grid_map_recenter_timing", options);
  GridMap map;
  map.initMap(node);
  // A dense boundary wall exercises inflation as well as the empty scan cost.
  std::vector<Eigen::Vector3d> wall;
  for (double y = -11.95; y < 12.0; y += 0.1)
    for (double z = -3.95; z < 4.0; z += 0.1)
      wall.emplace_back(11.95, y, z);
  for (int i = 0; i < 6; ++i) map.inputCloud(wall, Eigen::Vector3d(10.0, 0.0, 0.0));
  double total_ms = 0.0, max_ms = 0.0;
  for (int i = 0; i < 100; ++i) {
    const auto start = std::chrono::steady_clock::now();
    map.recenter(Eigen::Vector3d(i % 2 ? 0.0 : 0.1, 0.0, 0.0));
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    total_ms += ms;
    max_ms = std::max(max_ms, ms);
  }
  std::cout << "Default window one-cell recenter: mean " << total_ms / 100.0
            << " ms, max " << max_ms << " ms (100 shifts)" << std::endl;
  EXPECT_EQ(map.bufferCells(), 240u * 240u * 80u);
}

}  // namespace
