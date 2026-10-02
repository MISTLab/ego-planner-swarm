#include <gtest/gtest.h>
#include <ego_planner/ego_replan_fsm.h>
namespace ego_planner
{
class DoorEscape : public ::testing::Test {
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
        {"manager/feasibility_tolerance", 0.05},
        {"manager/planning_horizon", 7.5}, {"manager/drone_id", 0},
        {"manager/use_distinctive_trajs", true},
        {"optimization/lambda_smooth", 1.0}, {"optimization/lambda_collision", 0.5},
        {"optimization/lambda_feasibility", 0.1}, {"optimization/lambda_fitness", 1.0},
        {"optimization/dist0", 0.5}, {"optimization/swarm_clearance", 0.5},
        {"optimization/max_vel", 2.0}, {"optimization/max_acc", 3.0},
        {"bspline/limit_vel", 2.0}, {"bspline/limit_acc", 3.0},
        {"bspline/limit_ratio", 1.1}, {"prediction/obj_num", 0},
        {"grid_map/window_size_x", 12.0}, {"grid_map/window_size_y", 6.0},
        {"grid_map/window_size_z", 4.0}, {"grid_map/resolution", 0.1},
        {"fsm/report_occupied_start", true}, {"grid_map/obstacles_inflation", 0.3},
        {"grid_map/p_hit", 0.9}});
    node = std::make_shared<rclcpp::Node>("door_escape", options);
    fsm.init(node);
    fsm.have_odom_ = true;
    fsm.odom_pos_ = Eigen::Vector3d(-.306, .945, 1.769);
    fsm.odom_vel_.setZero();
    fsm.startup_published_ = true;
    fsm.planner_manager_->grid_map_->recenter(fsm.odom_pos_);
  }
  void pillar()
  {
    std::vector<Eigen::Vector3d> hits;
    for (double x = -.35; x < .67; x += .05) {
      for (double z = .05; z < 2.5; z += .05) {hits.push_back({x, 1.25, z});}}
    for (int i = 0; i < 5; ++i) {fsm.planner_manager_->grid_map_->inputCloud(hits, fsm.odom_pos_);}
  }
  rclcpp::Node::SharedPtr node;
  EGOReplanFSM fsm;
};
TEST_F(DoorEscape, R7InflatedButNotRawStartEscapesOutward) {
  pillar();
  auto map = fsm.planner_manager_->grid_map_;
  ASSERT_EQ(map->getOccupancy(fsm.odom_pos_), 0);
  ASSERT_EQ(map->getInflateOccupancy(fsm.odom_pos_), 1);
  fsm.planNextWaypoint({-.306, -1., 1.769});
  fsm.execFSMCallback();
  auto & info = fsm.planner_manager_->local_data_;
  auto start = info.position_traj_.evaluateDeBoorT(0);
  auto end = info.position_traj_.evaluateDeBoorT(info.duration_);
  EXPECT_TRUE(start.isApprox(fsm.odom_pos_));
  // First free cell is y=[.8,.9); target must be a full cell past its edge.
  EXPECT_LE(end.y(), .8);
  EXPECT_LE((end - start).norm(), .400001);
  EXPECT_EQ(map->getInflateOccupancy(end), 0);
  double previous = start.y();
  for (double t = .01; t <= info.duration_; t += .01) {
    auto p = info.position_traj_.evaluateDeBoorT(t);
    EXPECT_LE(p.y(), previous);
    EXPECT_EQ(map->getOccupancy(p), 0);
    previous = p.y();
  }
  fsm.checkCollisionCallback();
  EXPECT_NE(fsm.exec_state_, EGOReplanFSM::EMERGENCY_STOP);
}
TEST_F(DoorEscape, EscapeThenNormalPlanningOnceAndDriftDoesNotReescape) {
  pillar();
  const Eigen::Vector3d recorded_pose = fsm.odom_pos_;
  fsm.planNextWaypoint({-.306, -1., 1.769});
  fsm.execFSMCallback();
  ASSERT_EQ(fsm.exec_state_, EGOReplanFSM::INFLATED_ESCAPE);
  auto &info = fsm.planner_manager_->local_data_;
  fsm.odom_pos_ = info.position_traj_.evaluateDeBoorT(info.duration_);
  // Five cm of tracking error must not leave us inflated.
  fsm.odom_pos_.y() += .05;
  info.start_time_ = rclcpp::Clock().now() - rclcpp::Duration::from_seconds(4.);
  fsm.execFSMCallback();
  ASSERT_EQ(fsm.exec_state_, EGOReplanFSM::GEN_NEW_TRAJ);
  fsm.execFSMCallback();
  ASSERT_EQ(fsm.exec_state_, EGOReplanFSM::EXEC_TRAJ);
  fsm.odom_pos_ = recorded_pose;
  fsm.execFSMCallback();
  EXPECT_EQ(fsm.exec_state_, EGOReplanFSM::OCCUPIED_START);
  EXPECT_FALSE(fsm.have_target_);
}
TEST_F(DoorEscape, RawOccupiedStartStillYieldsToAdapter) {
  pillar();
  fsm.odom_pos_.y() = 1.25;
  fsm.planNextWaypoint({-.306, -1., 1.769});
  fsm.execFSMCallback();
  EXPECT_EQ(fsm.exec_state_, EGOReplanFSM::OCCUPIED_START);
}
TEST_F(DoorEscape, PlannerTrajectoryFits09mDoor) {
  auto manager = fsm.planner_manager_.get();
  auto map = manager->grid_map_;
  const Eigen::Vector3d start(-1.5, .05, 1.25), goal(1.5, .05, 1.25);
  map->recenter(start);
  std::vector<Eigen::Vector3d> hits;
  for (double y = -2.95; y < 3.; y += .1) {
    for (double z = .05; z < 3.; z += .1) {
      if (y < -.4 || y > .5 || z >= 2.) {hits.push_back({.05, y, z});}}
}
  for (int i = 0; i < 5; ++i) {
    map->inputCloud(hits, start);
}
  ASSERT_TRUE(manager->reboundReplan(start, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                                    goal, Eigen::Vector3d::Zero(), true, false));
  auto & info = manager->local_data_;
  for (double t = 0; t <= info.duration_; t += .01) {
    EXPECT_EQ(map->getInflateOccupancy(info.position_traj_.evaluateDeBoorT(t)), 0);
}
}

TEST_F(DoorEscape, AlongIntoRawAndUnboundedEscapeSegmentsAreRefused) {
  pillar();
  auto map = fsm.planner_manager_->grid_map_;
  const auto start = fsm.odom_pos_;
  EXPECT_TRUE(map->escapeSegmentSafe(start, start + Eigen::Vector3d(0, -.1, 0)));
  EXPECT_FALSE(map->escapeSegmentSafe(start, start + Eigen::Vector3d(.3, 0, 0)));
  EXPECT_FALSE(map->escapeSegmentSafe(start, start + Eigen::Vector3d(0, .35, 0)));
  EXPECT_FALSE(map->escapeSegmentSafe(start, start + Eigen::Vector3d(0, -.401, 0)));
  EXPECT_FALSE(map->escapeSegmentSafe(start, start));
}
TEST_F(DoorEscape, FlightBandCannotBeExemptedByEscape) {
  rclcpp::NodeOptions options;
  options.parameter_overrides({{"grid_map/flight_band_enabled", true},
      {"grid_map/flight_band_min", .8}, {"grid_map/flight_band_max", 1.5},
      {"grid_map/base_height", .125}, {"grid_map/resolution", .1},
      {"grid_map/obstacles_inflation", .3}});
  auto band_node = std::make_shared<rclcpp::Node>("escape_band", options);
  GridMap map;
  map.initMap(band_node);
  map.recenter({-.306, .945, .125}); // ground reference from landed odometry
  map.recenter(fsm.odom_pos_);
  Eigen::Vector3d end;
  EXPECT_FALSE(map.inflatedEscape(fsm.odom_pos_, fsm.odom_pos_ + Eigen::Vector3d(0, -1, 0), end));
}
TEST_F(DoorEscape, FailedTrackingYieldsToAdapterAfterOneAttempt) {
  pillar();
  fsm.planNextWaypoint({-.306, -1., 1.769});
  fsm.execFSMCallback();
  ASSERT_EQ(fsm.exec_state_, EGOReplanFSM::INFLATED_ESCAPE);
  auto & info = fsm.planner_manager_->local_data_;
  info.start_time_ = rclcpp::Clock().now() - rclcpp::Duration::from_seconds(4.);
  fsm.execFSMCallback();
  EXPECT_EQ(fsm.exec_state_, EGOReplanFSM::OCCUPIED_START);
  EXPECT_FALSE(fsm.have_target_);
}
TEST_F(DoorEscape, NewRawObstacleDuringEscapeStopsImmediately) {
  pillar();
  fsm.planNextWaypoint({-.306, -1., 1.769});
  fsm.execFSMCallback();
  ASSERT_EQ(fsm.exec_state_, EGOReplanFSM::INFLATED_ESCAPE);
  auto map = fsm.planner_manager_->grid_map_;
  for (int i = 0; i < 4; ++i) {
    map->inputCloud({fsm.odom_pos_}, fsm.odom_pos_ + Eigen::Vector3d(-1, 0, 0));
}
  fsm.checkCollisionCallback();
  EXPECT_EQ(fsm.exec_state_, EGOReplanFSM::OCCUPIED_START);
}
TEST_F(DoorEscape, MovingStartDoesNotAcquireAnUntrackableEscape) {
  pillar();
  fsm.odom_vel_ = Eigen::Vector3d(1, 0, 0);
  fsm.planNextWaypoint({-.306, -1., 1.769});
  fsm.execFSMCallback();
  EXPECT_EQ(fsm.exec_state_, EGOReplanFSM::OCCUPIED_START);
}
TEST_F(DoorEscape, GridInflationCloses06mDoor) {
  auto map = fsm.planner_manager_->grid_map_;
  std::vector<Eigen::Vector3d> hits;
  for (double y = -1.95; y < 2.; y += .1) {
    for (double z = .05; z < 3.; z += .1) {
      if (y < -.25 || y > .35 || z >= 2.) {hits.push_back({.05, y, z});}}
}
  for (int i = 0; i < 4; ++i) {
    map->inputCloud(hits, {-1., .05, 1.25});
}
  for (double y = -.25; y <= .35; y += .025) {
    EXPECT_EQ(map->getInflateOccupancy({.05, y, 1.25}), 1);
}
}

TEST_F(DoorEscape, PeerBodiesRemainBlockingDuringEscape) {
  pillar();
  auto manager = fsm.planner_manager_.get();
  manager->swarm_trajs_buf_.resize(2);
  auto & peer = manager->swarm_trajs_buf_[1];
  peer.drone_id = 1;
  peer.start_time_ = rclcpp::Clock().now();
  peer.duration_ = 10.;
  Eigen::MatrixXd controls(3, 6);
  for (int i = 0; i < 6; ++i) {
    controls.col(i) = fsm.odom_pos_ + Eigen::Vector3d(0, -.1, 0);
}
  peer.position_traj_ = UniformBspline(controls, 3, 4.);
  EXPECT_FALSE(manager->inflatedStartEscape(fsm.odom_pos_, {-.306, -1., 1.769}));
}

TEST_F(DoorEscape, CancelStopsTheEscapeSplineImmediately) {
  pillar();
  fsm.planNextWaypoint({-.306, -1., 1.769});
  fsm.execFSMCallback();
  ASSERT_EQ(fsm.exec_state_, EGOReplanFSM::INFLATED_ESCAPE);
  fsm.cancelCallback(nullptr);
  EXPECT_EQ(fsm.exec_state_, EGOReplanFSM::WAIT_TARGET);
  const auto controls = fsm.planner_manager_->local_data_.position_traj_.getControlPoint();
  for (int i = 0; i < controls.cols(); ++i) {
    EXPECT_TRUE(controls.col(i).isApprox(fsm.odom_pos_));
}
}
} // namespace ego_planner
