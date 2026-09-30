#include <gtest/gtest.h>
#include <atomic>
#include <future>
#include <thread>
#include <ego_planner/trajectory_publication_check.h>

namespace ego_planner
{
class TrajectoryPublicationCheck : public ::testing::Test
{
protected:
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}
  void SetUp() override
  {
    rclcpp::NodeOptions options;
    options.parameter_overrides({
        {"grid_map/window_size_x", 12.0}, {"grid_map/window_size_y", 4.0},
        {"grid_map/window_size_z", 4.0}, {"grid_map/resolution", 0.1},
        {"grid_map/obstacles_inflation", 0.2}, {"grid_map/p_hit", 0.9}});
    node = std::make_shared<rclcpp::Node>("publication_check", options);
    map.initMap(node);
    map.recenter({0, 0, 1.2});
  }
  UniformBspline straight()
  {
    Eigen::MatrixXd controls(3, 12);
    for (int i = 0; i < 12; ++i) {controls.col(i) = Eigen::Vector3d(i * .4, .05, 1.25);}
    return UniformBspline(controls, 3, .3);
  }
  void observe(const Eigen::Vector3d & point)
  {
    for (int i = 0; i < 4; ++i) {map.inputCloud({point}, {.05, .05, 1.25});}
  }
  rclcpp::Node::SharedPtr node;
  GridMap map;
};

TEST_F(TrajectoryPublicationCheck, RejectsObstacleFusedAfterCandidateWasPlanned)
{
  auto trajectory = straight();
  ASSERT_TRUE(trajectoryPublicationSafe(map, trajectory));
  // Sensing changes the map while the planner owns its private candidate.
  std::thread sensing([&]() {observe({1.65, .05, 1.25});});
  sensing.join();
  auto guard = map.lock();
  ASSERT_EQ(map.getInflateOccupancy({1.65, .05, 1.25}), 1);
  EXPECT_FALSE(trajectoryPublicationSafe(map, trajectory));
}

TEST_F(TrajectoryPublicationCheck, RetainsUpstreamUncheckedTailPolicy)
{
  auto trajectory = straight();
  observe({3.85, .05, 1.25});
  auto guard = map.lock();
  ASSERT_EQ(map.getInflateOccupancy({3.85, .05, 1.25}), 1);
  EXPECT_TRUE(trajectoryPublicationSafe(map, trajectory));
}

TEST_F(TrajectoryPublicationCheck, FusionCannotChangeMapBetweenFinalCheckAndPublish)
{
  auto trajectory = straight();
  auto guard = map.lock();
  std::promise<void> started;
  auto finished = std::async(std::launch::async, [&]() {
        started.set_value();
        observe({1.65, .05, 1.25});
  });
  started.get_future().wait();
  EXPECT_TRUE(trajectoryPublicationSafe(map, trajectory));
  EXPECT_EQ(finished.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
  guard.unlock();
  finished.get();
  EXPECT_FALSE(trajectoryPublicationSafe(map, trajectory));
}
} // namespace ego_planner
