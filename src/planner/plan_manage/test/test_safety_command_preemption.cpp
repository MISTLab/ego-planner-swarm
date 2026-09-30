#include <gtest/gtest.h>
#include <chrono>
#include <functional>
#include <thread>
#include <ego_planner/ego_replan_fsm.h>
#include <ego_planner/trajectory_publication_check.h>

namespace ego_planner
{
namespace
{
// Replace only the expensive optimizer boundary at link time. Its real side
// effect on success is updateTrajInfo; everything after it (gate, rollback,
// FSM transitions and ROS publication) is production code. The hook injects
// the exact interleaving that a separate command/sensing executor can cause.
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

class SafetyCommandPreemption : public ::testing::Test
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
        {"manager/max_vel", 2.0}, {"manager/max_acc", 3.0},
        {"manager/max_jerk", 4.0}, {"manager/control_points_distance", 0.4},
        {"manager/planning_horizon", 7.5}, {"manager/drone_id", 0},
        {"optimization/swarm_clearance", 0.5},
        {"grid_map/window_size_x", 12.0}, {"grid_map/window_size_y", 6.0},
        {"grid_map/window_size_z", 4.0}, {"grid_map/resolution", 0.1},
        {"grid_map/obstacles_inflation", 0.1}, {"grid_map/p_hit", 0.9}});
    node = std::make_shared<rclcpp::Node>("safety_preemption", options);
    fsm.init(node);
    // No planning executor is spun: the test calls the real safety callback
    // once, then inspects what it published BEFORE any exec tick can run.
    observer = std::make_shared<rclcpp::Node>("safety_preemption_observer");
    subscription = observer->create_subscription<traj_utils::msg::Bspline>(
      "planning/bspline", 10,
      [this](traj_utils::msg::Bspline::ConstSharedPtr msg) {published.push_back(*msg);});
    state_subscription = observer->create_subscription<traj_utils::msg::CommandState>(
      "planning/command_state", rclcpp::QoS(100).transient_local(),
      [this](traj_utils::msg::CommandState::ConstSharedPtr msg) {states.push_back(*msg);});
    observer_executor.add_node(observer);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (fsm.bspline_pub_->get_subscription_count() == 0 &&
      std::chrono::steady_clock::now() < deadline)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_GT(fsm.bspline_pub_->get_subscription_count(), 0u);
    fsm.have_odom_ = true;
    fsm.odom_pos_ = Eigen::Vector3d(.4, .05, 1.25);
    fsm.odom_vel_.setZero();
    fsm.startup_published_ = true;
    fsm.planner_manager_->grid_map_->recenter(fsm.odom_pos_);
    fsm.planNextWaypoint(Eigen::Vector3d(4.0, .05, 1.25));
    fsm.planner_manager_->updateTrajInfo(straight(.05), rclcpp::Clock().now());
    fsm.exec_state_ = EGOReplanFSM::EXEC_TRAJ;
    fsm.start_pt_ = fsm.odom_pos_;
    fsm.start_vel_.setZero();
    fsm.start_acc_.setZero();
    previous = fsm.planner_manager_->local_data_;
  }

  void TearDown() override {optimizer_result = nullptr;}

  UniformBspline straight(double y)
  {
    Eigen::MatrixXd controls(3, 12);
    for (int i = 0; i < 12; ++i) {controls.col(i) = Eigen::Vector3d(i * .4, y, 1.25);}
    return UniformBspline(controls, 3, .3);
  }

  void queueGoal()
  {
    traj_utils::msg::PlannerCommand command;
    command.sequence = fsm.ordered_commands_ ? 1 : 0;
    command.command = traj_utils::msg::PlannerCommand::GOAL;
    command.goal.pose.position.x = -5.0;
    command.goal.pose.position.z = 1.25;
    fsm.receiveCommand(command);
  }

  void observe(const Eigen::Vector3d & point)
  {
    for (int i = 0; i < 4; ++i) {
      fsm.planner_manager_->grid_map_->inputCloud({point}, fsm.odom_pos_);
    }
  }

  void collectPublications()
  {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
    do {
      observer_executor.spin_some();
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    } while (std::chrono::steady_clock::now() < deadline);
  }

  void expectPreviousTrajectory()
  {
    const auto & current = fsm.planner_manager_->local_data_;
    EXPECT_EQ(current.traj_id_, previous.traj_id_);
    EXPECT_EQ(current.start_time_, previous.start_time_);
    EXPECT_EQ(current.duration_, previous.duration_);
    EXPECT_TRUE(current.start_pos_.isApprox(previous.start_pos_));
    auto position = current.position_traj_;
    auto velocity = current.velocity_traj_;
    auto acceleration = current.acceleration_traj_;
    EXPECT_TRUE(position.getControlPoint().isApprox(previous.position_traj_.getControlPoint()));
    EXPECT_TRUE(position.getKnot().isApprox(previous.position_traj_.getKnot()));
    EXPECT_TRUE(velocity.getControlPoint().isApprox(previous.velocity_traj_.getControlPoint()));
    EXPECT_TRUE(acceleration.getControlPoint().isApprox(
        previous.acceleration_traj_.getControlPoint()));
  }

  void expireLidar()
  {
    auto & map = *fsm.planner_manager_->grid_map_;
    auto guard = map.lock();
    map.inputCloud({}, fsm.odom_pos_);
    map.md_.last_cloud_ -= std::chrono::seconds(10);
    map.checkSensorTimeout();
    ASSERT_TRUE(map.getOdomDepthTimeout());
  }

  void expectTimeoutHoldAndRecovery(bool ordered, bool queue_before_stop = true)
  {
    fsm.ordered_commands_ = ordered;
    int attempts = 0;
    optimizer_result = [&](EGOPlannerManager & manager) {
        ++attempts;
        manager.updateTrajInfo(straight(1.05), rclcpp::Clock().now());
        return true;
      };
    expireLidar();
    if (queue_before_stop) {queueGoal();}
    fsm.checkCollisionCallback();
    collectPublications();
    ASSERT_EQ(published.size(), 1u) << "lidar-timeout stop was not published immediately";
    if (!queue_before_stop) {
      queueGoal();
      fsm.drainInputs();
      collectPublications();
      EXPECT_EQ(published.size(), 1u) << "queued goal replaced the existing stop";
    }
    EXPECT_EQ(fsm.command_sequence_, ordered ? 1u : 0u);
    EXPECT_EQ(fsm.goal_sequence_, ordered ? 1u : 0u);
    ASSERT_FALSE(states.empty());
    EXPECT_EQ(states.back().command_sequence, ordered ? 1u : 0u);
    EXPECT_EQ(states.back().goal_sequence, ordered ? 1u : 0u);
    EXPECT_EQ(states.back().state, "EMERGENCY_STOP");
    EXPECT_EQ(fsm.exec_state_, EGOReplanFSM::EMERGENCY_STOP);
    EXPECT_TRUE(fsm.have_pending_goal_);
    EXPECT_DOUBLE_EQ(fsm.pending_goal_.x(), -5.0);
    for (const auto & point : published.front().pos_pts) {
      EXPECT_DOUBLE_EQ(point.x, fsm.odom_pos_.x());
      EXPECT_DOUBLE_EQ(point.y, fsm.odom_pos_.y());
      EXPECT_DOUBLE_EQ(point.z, fsm.odom_pos_.z());
    }
    for (int i = 0; i < 3; ++i) {
      fsm.execFSMCallback();
      fsm.checkCollisionCallback();
    }
    collectPublications();
    EXPECT_EQ(attempts, 0) << "planning ran on a timed-out map";
    EXPECT_EQ(published.size(), 1u) << "new trajectory or repeated stop while timed out";
    for (const auto & state : states) {EXPECT_EQ(state.state, "EMERGENCY_STOP");}
    fsm.planner_manager_->grid_map_->inputCloud({Eigen::Vector3d(5.05, 2.05, 1.25)}, fsm.odom_pos_);
    ASSERT_FALSE(fsm.planner_manager_->grid_map_->getOdomDepthTimeout());
    fsm.execFSMCallback();
    collectPublications();
    EXPECT_EQ(attempts, 1);
    ASSERT_EQ(published.size(), 2u) << "pending goal did not resume after fresh sensing";
    EXPECT_EQ(fsm.exec_state_, EGOReplanFSM::EXEC_TRAJ);
    EXPECT_FALSE(fsm.have_pending_goal_);
    EXPECT_DOUBLE_EQ(fsm.end_pt_.x(), -5.0);
  }

  rclcpp::Node::SharedPtr node, observer;
  EGOReplanFSM fsm;
  LocalTrajData previous;
  rclcpp::executors::SingleThreadedExecutor observer_executor;
  rclcpp::Subscription<traj_utils::msg::Bspline>::SharedPtr subscription;
  std::vector<traj_utils::msg::Bspline> published;
  rclcpp::Subscription<traj_utils::msg::CommandState>::SharedPtr state_subscription;
  std::vector<traj_utils::msg::CommandState> states;
};

TEST_F(SafetyCommandPreemption, QueuedGoalCannotDiscardSuccessfulSafetyReplan)
{
  // The old trajectory hits this voxel within emergency_time_ (about 0.3 s).
  observe({.85, .05, 1.25});
  int attempts = 0;
  optimizer_result = [&](EGOPlannerManager & manager) {
      ++attempts;
      manager.updateTrajInfo(straight(1.05), rclcpp::Clock().now());
      queueGoal();
      return true;
    };
  fsm.checkCollisionCallback();
  collectPublications();
  ASSERT_EQ(published.size(), 1u) << "safety callback silently continued the colliding trajectory";
  EXPECT_EQ(attempts, 1);
  EXPECT_EQ(fsm.exec_state_, EGOReplanFSM::EXEC_TRAJ);
  EXPECT_GT(published.front().traj_id, previous.traj_id_);
  EXPECT_DOUBLE_EQ(published.front().pos_pts.front().y, 1.05);
  auto guard = fsm.planner_manager_->grid_map_->lock();
  EXPECT_TRUE(trajectoryPublicationSafe(
      *fsm.planner_manager_->grid_map_, fsm.planner_manager_->local_data_.position_traj_));
  EXPECT_TRUE(fsm.commandsPending());
  fsm.drainInputs();
  EXPECT_EQ(fsm.command_sequence_, 1u);
  EXPECT_EQ(fsm.exec_state_, EGOReplanFSM::REPLAN_TRAJ);
}

TEST_F(SafetyCommandPreemption, FailedSafetyReplanPublishesStopBeforeQueuedGoal)
{
  observe({.85, .05, 1.25});
  int attempts = 0;
  optimizer_result = [&](EGOPlannerManager &) {
      ++attempts;
      queueGoal();
      return false;
    };
  fsm.checkCollisionCallback();
  collectPublications();
  ASSERT_EQ(published.size(), 1u) << "emergency stop was deferred until after queued commands";
  EXPECT_EQ(attempts, 3) << "a safety retry was preempted";
  EXPECT_EQ(fsm.exec_state_, EGOReplanFSM::EMERGENCY_STOP);
  for (const auto & point : published.front().pos_pts) {
    EXPECT_DOUBLE_EQ(point.x, fsm.odom_pos_.x());
    EXPECT_DOUBLE_EQ(point.y, fsm.odom_pos_.y());
    EXPECT_DOUBLE_EQ(point.z, fsm.odom_pos_.z());
  }
  const auto stopped = fsm.planner_manager_->local_data_;
  fsm.drainInputs();
  EXPECT_EQ(fsm.command_sequence_, 1u);
  EXPECT_EQ(fsm.exec_state_, EGOReplanFSM::GEN_NEW_TRAJ);
  EXPECT_EQ(fsm.planner_manager_->local_data_.traj_id_, stopped.traj_id_);
}

TEST_F(SafetyCommandPreemption, PersistentCollisionKeepsOriginalStopWithoutRepublishing)
{
  observe({.45, .05, 1.25});
  ASSERT_EQ(fsm.planner_manager_->grid_map_->getInflateOccupancy(fsm.odom_pos_), 1);
  int attempts = 0;
  optimizer_result = [&](EGOPlannerManager &) {++attempts; return false;};
  fsm.checkCollisionCallback();
  collectPublications();
  ASSERT_EQ(published.size(), 1u);
  ASSERT_EQ(fsm.exec_state_, EGOReplanFSM::EMERGENCY_STOP);
  ASSERT_FALSE(fsm.flag_escape_emergency_);
  previous = fsm.planner_manager_->local_data_;

  // Momentum changes odometry, but a persistent collision at occupied P0
  // must not make the hold point follow it on the next safety tick.
  fsm.odom_pos_.x() += .05;
  fsm.checkCollisionCallback();
  collectPublications();
  EXPECT_EQ(attempts, 6) << "both safety callbacks must exercise the failing fallback";
  EXPECT_EQ(published.size(), 1u) << "already-published stop was sent again";
  expectPreviousTrajectory();
}

TEST_F(SafetyCommandPreemption, GateRejectionRestoresEntirePreviousTrajectory)
{
  optimizer_result = [&](EGOPlannerManager & manager) {
      manager.updateTrajInfo(straight(1.05), rclcpp::Clock().now());
      // This obstacle arrives AFTER the optimizer produced its candidate.
      observe({1.65, 1.05, 1.25});
      return true;
    };
  EXPECT_FALSE(fsm.callReboundReplan(false, false));
  collectPublications();
  EXPECT_TRUE(published.empty());
  expectPreviousTrajectory();
}

TEST_F(SafetyCommandPreemption, PendingCommandDiscardsOrdinaryCandidateAndRestoresPrevious)
{
  optimizer_result = [&](EGOPlannerManager & manager) {
      manager.updateTrajInfo(straight(1.05), rclcpp::Clock().now());
      queueGoal();
      return true;
    };
  EXPECT_FALSE(fsm.callReboundReplan(false, false));
  collectPublications();
  EXPECT_TRUE(published.empty());
  expectPreviousTrajectory();
  EXPECT_TRUE(fsm.commandsPending());
}
TEST_F(SafetyCommandPreemption, RejectedSafetyCandidateStopsDespitePendingGoal)
{
  observe({.85, .05, 1.25});
  int attempts = 0;
  optimizer_result = [&](EGOPlannerManager & manager) {
      ++attempts;
      manager.updateTrajInfo(straight(1.05), rclcpp::Clock().now());
      observe({1.65, 1.05, 1.25});
      queueGoal();
      return true;
    };
  fsm.checkCollisionCallback();
  collectPublications();
  ASSERT_EQ(published.size(), 1u);
  EXPECT_EQ(attempts, 3);
  EXPECT_EQ(fsm.exec_state_, EGOReplanFSM::EMERGENCY_STOP);
  for (const auto & point : published.front().pos_pts) {
    EXPECT_DOUBLE_EQ(point.x, fsm.odom_pos_.x());
    EXPECT_DOUBLE_EQ(point.y, fsm.odom_pos_.y());
    EXPECT_DOUBLE_EQ(point.z, fsm.odom_pos_.z());
  }
}

TEST_F(SafetyCommandPreemption, LidarTimeoutStopsImmediatelyWithoutContinuingCollisionReplan)
{
  observe({.85, .05, 1.25});
  expireLidar();
  int attempts = 0;
  optimizer_result = [&](EGOPlannerManager &) {++attempts; return false;};
  fsm.checkCollisionCallback();
  collectPublications();
  ASSERT_EQ(published.size(), 1u);
  EXPECT_EQ(attempts, 0);
  EXPECT_EQ(fsm.exec_state_, EGOReplanFSM::EMERGENCY_STOP);
  EXPECT_FALSE(fsm.flag_escape_emergency_);
  fsm.execFSMCallback();
  collectPublications();
  EXPECT_EQ(published.size(), 1u);
}

TEST_F(SafetyCommandPreemption, OrderedGoalWaitsForFreshLidarAfterImmediateStop)
{
  expectTimeoutHoldAndRecovery(true);
}

TEST_F(SafetyCommandPreemption, LegacyGoalWaitsForFreshLidarAfterImmediateStop)
{
  expectTimeoutHoldAndRecovery(false);
}

TEST_F(SafetyCommandPreemption, QueuedGoalCannotLeaveExistingLidarStop)
{
  expectTimeoutHoldAndRecovery(true, false);
}

TEST_F(SafetyCommandPreemption, LidarTimeoutDuringOptimizationRejectsCandidate)
{
  optimizer_result = [&](EGOPlannerManager & manager) {
      manager.updateTrajInfo(straight(1.05), rclcpp::Clock().now());
      expireLidar();
      return true;
    };
  EXPECT_FALSE(fsm.callReboundReplan(false, false));
  collectPublications();
  EXPECT_TRUE(published.empty());
  expectPreviousTrajectory();
}
} // namespace ego_planner
