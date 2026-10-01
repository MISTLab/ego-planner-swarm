#include <gtest/gtest.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <new>
#include <string>

#include <ego_planner/ego_replan_fsm.h>

// Every operator-new block of this test binary is preceded by 32 bytes of
// doubles equal to kHeapGarbage. A read just before a std::vector's storage
// then sees the same large finite value in every run, where the field process
// saw whatever its heap held (drone-r5 replan 8399).
namespace
{
constexpr std::size_t kPrefixBytes = 32;
constexpr double kHeapGarbage = 1e9;
}  // namespace

void * operator new(std::size_t size)
{
  void * block = std::malloc(size + kPrefixBytes);
  if (block == nullptr) {throw std::bad_alloc();}
  double * prefix = static_cast<double *>(block);
  for (std::size_t i = 0; i < kPrefixBytes / sizeof(double); ++i) {
    prefix[i] = kHeapGarbage;
  }
  return static_cast<char *>(block) + kPrefixBytes;
}

void operator delete(void * memory) noexcept
{
  if (memory != nullptr) {std::free(static_cast<char *>(memory) - kPrefixBytes);}
}

void operator delete(void * memory, std::size_t) noexcept {::operator delete(memory);}

namespace ego_planner
{
namespace
{
// Death-test child exit codes.
constexpr int kPlanned = 0;
constexpr int kRejected = 3;
constexpr int kHeapBudgetExceeded = 42;

// A bounded replan needs a few MiB; the unbounded ones reach the budget in
// well under a second and would otherwise take the host's memory.
constexpr std::size_t kHeapBudgetBytes = 256u << 20;
constexpr unsigned kChildSeconds = 20;

// Cap the heap (brk and private mappings) at its current size plus the budget,
// so a runaway allocation throws std::bad_alloc in this death-test child; an
// endless loop is killed by SIGALRM. Both are failures of the planner.
void boundThisProcess()
{
  std::ifstream status("/proc/self/status");
  std::string line;
  rlim_t data_bytes = 0;
  while (std::getline(status, line)) {
    if (line.rfind("VmData:", 0) == 0) {data_bytes = std::stoull(line.substr(7)) * 1024;}
  }
  rlimit limit;
  limit.rlim_cur = limit.rlim_max = data_bytes + kHeapBudgetBytes;
  if (data_bytes == 0 || setrlimit(RLIMIT_DATA, &limit) != 0) {_exit(2);}
  alarm(kChildSeconds);
}

[[noreturn]] void exitWithReplanOutcome(const std::function<bool()> & replan)
{
  boundThisProcess();
  try {
    _exit(replan() ? kPlanned : kRejected);
  } catch (const std::bad_alloc &) {
    _exit(kHeapBudgetExceeded);
  }
}

bool exitedCleanly(int status)
{
  return WIFEXITED(status) &&
         (WEXITSTATUS(status) == kPlanned || WEXITSTATUS(status) == kRejected);
}
}  // namespace

class PlanningBounds : public ::testing::Test
{
protected:
  static void SetUpTestSuite()
  {
    // Re-execute for each death test: the FSM's node owns ROS threads.
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    rclcpp::init(0, nullptr);
  }
  static void TearDownTestSuite() {rclcpp::shutdown();}

  void SetUp() override
  {
    // SwarmDeck's drone parameters (deploy/drone/drone.launch.py), on a
    // smaller map and without the flight band.
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
        {"grid_map/obstacles_inflation", 0.25}, {"grid_map/p_hit", 0.65}});
    node = std::make_shared<rclcpp::Node>("planning_bounds", options);
    fsm.init(node);
    fsm.have_odom_ = true;
    fsm.odom_pos_ = Eigen::Vector3d(.4, .05, 1.25);
    fsm.odom_vel_.setZero();
    fsm.startup_published_ = true;
    fsm.planner_manager_->grid_map_->recenter(fsm.odom_pos_);
    fsm.planNextWaypoint(Eigen::Vector3d(5.0, .05, 1.25));
    // The process's first replan always starts from a polynomial (a static
    // flag in reboundReplan); the field node was thousands of replans in.
    ASSERT_TRUE(manager().reboundReplan(
        fsm.odom_pos_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d(3.0, .05, 1.25), Eigen::Vector3d::Zero(), true, false));
  }

  EGOPlannerManager & manager() {return *fsm.planner_manager_;}

  // 8 control points 0.4 m apart: a 1.5 s trajectory ending near x = 2.4.
  UniformBspline straight()
  {
    Eigen::MatrixXd controls(3, 8);
    for (int i = 0; i < 8; ++i) {controls.col(i) = Eigen::Vector3d(i * .4, .05, 1.25);}
    return UniformBspline(controls, 3, .3);
  }

  // REPLAN_TRAJ's first attempt (planFromCurrentTraj, then callReboundReplan
  // with neither a new target nor a polynomial initialization).
  bool replanFromCurrentTrajectory()
  {
    auto & info = manager().local_data_;
    const double t_cur = (rclcpp::Clock().now() - info.start_time_).seconds();
    fsm.start_pt_ = info.position_traj_.evaluateDeBoorT(t_cur);
    fsm.start_vel_ = info.velocity_traj_.evaluateDeBoorT(t_cur);
    fsm.start_acc_ = info.acceleration_traj_.evaluateDeBoorT(t_cur);
    fsm.getLocalTarget();
    return manager().reboundReplan(
      fsm.start_pt_, fsm.start_vel_, fsm.start_acc_, fsm.local_target_pt_,
      fsm.local_target_vel_, false, false);
  }

  rclcpp::Node::SharedPtr node;
  EGOReplanFSM fsm;
};

TEST_F(PlanningBounds, ReplanAfterTheFlownTrajectoryEndedStaysWithinHeapBudget)
{
  // drone-r5: REPLAN_TRAJ failed for longer than the rest of the flown
  // trajectory. The next replan sampled none of it, and indexed the empty
  // sample list's predecessor: its arc length came from unowned heap bytes.
  manager().updateTrajInfo(straight(), rclcpp::Clock().now() - rclcpp::Duration::from_seconds(60));
  EXPECT_EXIT(
    exitWithReplanOutcome([this] {return replanFromCurrentTrajectory();}),
    ::testing::ExitedWithCode(kPlanned), "");
}

TEST_F(PlanningBounds, ReplanWithOneSampleLeftAndANearTargetTerminates)
{
  // One sample remains of the flown trajectory and the target is within one
  // time step of it: the arc-length list has a single entry.
  const auto trajectory = straight();
  auto copy = trajectory;
  const double duration = copy.getTimeSum();
  manager().updateTrajInfo(
    trajectory, rclcpp::Clock().now() - rclcpp::Duration::from_seconds(duration - .1));
  EXPECT_EXIT(
    exitWithReplanOutcome([this] {
      auto & info = manager().local_data_;
      const double t_cur = (rclcpp::Clock().now() - info.start_time_).seconds();
      const Eigen::Vector3d start = info.position_traj_.evaluateDeBoorT(t_cur);
      return manager().reboundReplan(
        start, info.velocity_traj_.evaluateDeBoorT(t_cur),
        info.acceleration_traj_.evaluateDeBoorT(t_cur),
        start + Eigen::Vector3d(.25, 0, 0), Eigen::Vector3d::Zero(), false, false);
    }),
    exitedCleanly, "");
}

TEST_F(PlanningBounds, LargeStartVelocityCannotGrowTheInitialPath)
{
  // The polynomial initialization samples until consecutive points are
  // within 1.5 control-point distances; parameterizeToBspline then solves a
  // dense (K + 4) x (K + 2) system. 10 km/s needs K ~ 40000: about 13 GB.
  EXPECT_EXIT(
    exitWithReplanOutcome([this] {
      return manager().reboundReplan(
        fsm.odom_pos_, Eigen::Vector3d(1e4, 0, 0), Eigen::Vector3d::Zero(),
        Eigen::Vector3d(3.0, .05, 1.25), Eigen::Vector3d::Zero(), true, false);
    }),
    ::testing::ExitedWithCode(kRejected), "");
}

TEST_F(PlanningBounds, NonFiniteReplanInputsAreRejected)
{
  EXPECT_EXIT(
    exitWithReplanOutcome([this] {
      const double nan = std::numeric_limits<double>::quiet_NaN();
      const double inf = std::numeric_limits<double>::infinity();
      const Eigen::Vector3d zero = Eigen::Vector3d::Zero(), target(3.0, .05, 1.25);
      bool planned = false;
      for (bool poly_init : {true, false}) {
        planned |= manager().reboundReplan(
          Eigen::Vector3d(nan, .05, 1.25), zero, zero, target, zero, poly_init, false);
        planned |= manager().reboundReplan(
          fsm.odom_pos_, Eigen::Vector3d(inf, 0, 0), zero, target, zero, poly_init, false);
        planned |= manager().reboundReplan(
          fsm.odom_pos_, zero, Eigen::Vector3d(0, nan, 0), target, zero, poly_init, false);
        planned |= manager().reboundReplan(
          fsm.odom_pos_, zero, zero, Eigen::Vector3d(3.0, inf, 1.25), zero, poly_init, false);
        planned |= manager().reboundReplan(
          fsm.odom_pos_, zero, zero, target, Eigen::Vector3d(0, 0, nan), poly_init, false);
      }
      return planned;
    }),
    ::testing::ExitedWithCode(kRejected), "");
}

TEST_F(PlanningBounds, TimeReallocationRejectsNonFiniteAndExcessiveRatios)
{
  // reparamBspline stretches the knots by the ratio and resamples every dt:
  // an infinite ratio gives an infinite dt and an endless sampling loop, a
  // NaN one an empty resampling. 1e6 is finite and bounded in memory; it is
  // rejected by policy (kMaxTimeReallocationRatio), not as a memory bug.
  for (double ratio : {std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN(), 1e6})
  {
    EXPECT_EXIT(
      exitWithReplanOutcome([this, ratio] {
        UniformBspline trajectory = straight();
        std::vector<Eigen::Vector3d> derivatives(4, Eigen::Vector3d::Zero());
        double ts = .3;
        Eigen::MatrixXd refined;
        return manager().refineTrajAlgo(trajectory, derivatives, ratio, ts, refined);
      }),
      ::testing::ExitedWithCode(kRejected), "") << "ratio " << ratio;
  }
}

TEST_F(PlanningBounds, OrdinaryReplansStillPlan)
{
  // A live trajectory, as in every normal EXEC_TRAJ replan.
  manager().updateTrajInfo(straight(), rclcpp::Clock().now());
  EXPECT_TRUE(replanFromCurrentTrajectory());
  EXPECT_TRUE(manager().reboundReplan(
    fsm.odom_pos_, Eigen::Vector3d(1.0, 0, 0), Eigen::Vector3d::Zero(),
    Eigen::Vector3d(4.0, .05, 1.25), Eigen::Vector3d::Zero(), true, false));
  EXPECT_TRUE(manager().reboundReplan(
    fsm.odom_pos_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
    Eigen::Vector3d(4.0, 1.05, 1.25), Eigen::Vector3d::Zero(), true, true));
}
}  // namespace ego_planner
