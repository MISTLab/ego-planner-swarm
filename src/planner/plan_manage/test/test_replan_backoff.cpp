#include <gtest/gtest.h>
#include <chrono>
#include <functional>
#include <thread>
#include <ego_planner/ego_replan_fsm.h>

namespace ego_planner
{
namespace
{
// Replace the optimizer at link time (as test_safety_command_preemption does):
// these tests count how often the real FSM calls it.
std::function<bool(EGOPlannerManager &)> optimizer_result;
}

extern "C" bool wrappedReboundReplan(
  EGOPlannerManager *, Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d,
  Eigen::Vector3d, Eigen::Vector3d, bool, bool)
asm (
  "__wrap__ZN11ego_planner17EGOPlannerManager13reboundReplanEN5Eigen6MatrixIdLi3ELi1ELi0ELi3ELi1EEES3_S3_S3_S3_bb");

extern "C" bool wrappedReboundReplan(
  EGOPlannerManager * manager, Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d,
  Eigen::Vector3d, Eigen::Vector3d, bool, bool)
{
  return optimizer_result(*manager);
}

class ReplanBackoff : public ::testing::Test
{
protected:
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}

  void SetUp() override
  {
    rclcpp::NodeOptions options;
    options.parameter_overrides({
        {"fsm/flight_type", 1}, {"fsm/ordered_commands", true},
        {"fsm/planning_horizon", 7.5}, {"fsm/emergency_time", 1.0},
        {"fsm/thresh_replan_time", 1.0}, {"fsm/thresh_no_replan_meter", 1.0},
        {"manager/max_vel", 2.0}, {"manager/max_acc", 3.0},
        {"manager/max_jerk", 4.0}, {"manager/control_points_distance", 0.4},
        {"manager/planning_horizon", 7.5}, {"manager/drone_id", 0},
        {"optimization/swarm_clearance", 0.5},
        {"grid_map/window_size_x", 12.0}, {"grid_map/window_size_y", 6.0},
        {"grid_map/window_size_z", 4.0}, {"grid_map/resolution", 0.1},
        {"grid_map/obstacles_inflation", 0.1}, {"grid_map/p_hit", 0.9}});
    node = std::make_shared<rclcpp::Node>("replan_backoff", options);
    fsm.init(node);
    fsm.have_odom_ = true;
    fsm.odom_pos_ = Eigen::Vector3d(.4, .05, 1.25);
    fsm.odom_vel_.setZero();
    fsm.startup_published_ = true;
    fsm.planner_manager_->grid_map_->recenter(fsm.odom_pos_);
    fsm.planNextWaypoint(Eigen::Vector3d(4.0, .05, 1.25));
    fsm.planner_manager_->updateTrajInfo(straight(), rclcpp::Clock().now());
    fsm.changeFSMExecState(EGOReplanFSM::EXEC_TRAJ, "TEST");
    fsm.changeFSMExecState(EGOReplanFSM::REPLAN_TRAJ, "TEST");
  }

  void TearDown() override {optimizer_result = nullptr;}

  UniformBspline straight()
  {
    Eigen::MatrixXd controls(3, 12);
    for (int i = 0; i < 12; ++i) {controls.col(i) = Eigen::Vector3d(i * .4, .05, 1.25);}
    return UniformBspline(controls, 3, .3);
  }

  // Run the FSM like its 10 ms timer for `duration`.
  void tick(std::chrono::milliseconds duration)
  {
    const auto end = std::chrono::steady_clock::now() + duration;
    while (std::chrono::steady_clock::now() < end) {
      fsm.execFSMCallback();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  rclcpp::Node::SharedPtr node;
  EGOReplanFSM fsm;
};

TEST_F(ReplanBackoff, FailingReplanBacksOffInsteadOfRetryingEveryTick)
{
  int attempts = 0;
  optimizer_result = [&](EGOPlannerManager &) {++attempts; return false;};
  tick(std::chrono::milliseconds(1000));
  // Every 10 ms tick makes three attempts: about 300 in a second without a
  // back-off; with it, seven ticks (immediate, immediate, then 20 ms doubling).
  EXPECT_GE(attempts, 15) << "a failing replan must still be retried";
  EXPECT_LE(attempts, 30);
  EXPECT_EQ(fsm.exec_state_, EGOReplanFSM::REPLAN_TRAJ);

  // The back-off is capped at 0.5 s: retries go on at about 2 a second
  // (4 or 5 ticks of 3 attempts in 2 s, plus one for timer slack).
  attempts = 0;
  tick(std::chrono::milliseconds(2000));
  EXPECT_GE(attempts, 9);
  EXPECT_LE(attempts, 18);
}

TEST_F(ReplanBackoff, AnotherStateEndsTheBackoff)
{
  int attempts = 0;
  optimizer_result = [&](EGOPlannerManager &) {++attempts; return false;};
  tick(std::chrono::milliseconds(700));
  ASSERT_GE(attempts, 15);

  // A successful replan is published and executed.
  optimizer_result = [&](EGOPlannerManager & manager) {
      ++attempts;
      manager.updateTrajInfo(straight(), rclcpp::Clock().now());
      return true;
    };
  tick(std::chrono::milliseconds(600));
  ASSERT_EQ(fsm.exec_state_, EGOReplanFSM::EXEC_TRAJ);

  // The next REPLAN_TRAJ (here: a new goal while executing) plans at once.
  attempts = 0;
  optimizer_result = [&](EGOPlannerManager &) {++attempts; return false;};
  fsm.planNextWaypoint(Eigen::Vector3d(4.0, 1.05, 1.25));
  ASSERT_EQ(fsm.exec_state_, EGOReplanFSM::REPLAN_TRAJ);
  fsm.execFSMCallback();
  EXPECT_EQ(attempts, 3);
}
}  // namespace ego_planner
