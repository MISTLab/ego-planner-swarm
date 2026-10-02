#ifndef _REBO_REPLAN_FSM_H_
#define _REBO_REPLAN_FSM_H_

#include <Eigen/Eigen>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <deque>
#include <mutex>
#include "traj_utils/msg/planner_command.hpp"
#include "traj_utils/msg/command_state.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/empty.hpp"
#include "std_msgs/msg/string.hpp"
#include <vector>
#include "visualization_msgs/msg/marker.hpp"

#include "bspline_opt/bspline_optimizer.h"
#include "plan_env/grid_map.h"
#include "traj_utils/msg/bspline.hpp"
#include "traj_utils/msg/multi_bsplines.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "traj_utils/msg/data_disp.hpp"
#include "ego_planner/planner_manager.h"
#include "traj_utils/planning_visualization.h"

using std::vector;

namespace ego_planner
{

  class EGOReplanFSM
  {

  private:
    /* ---------- flag ---------- */
    enum FSM_EXEC_STATE
    {
      INIT,
      WAIT_TARGET,
      GEN_NEW_TRAJ,
      REPLAN_TRAJ,
      EXEC_TRAJ,
      EMERGENCY_STOP,
      SEQUENTIAL_START,
      OCCUPIED_START,
      INFLATED_ESCAPE
    };
    enum TARGET_TYPE
    {
      MANUAL_TARGET = 1,
      PRESET_TARGET = 2,
      REFENCE_PATH = 3
    };

    /* planning utils */
    EGOPlannerManager::Ptr planner_manager_;
    PlanningVisualization::Ptr visualization_;
    traj_utils::msg::DataDisp data_disp_;
    traj_utils::msg::MultiBsplines multi_bspline_msgs_buf_;

    /* parameters */
    int target_type_; // 1 mannual select, 2 hard code
    double no_replan_thresh_, replan_thresh_;
    double waypoints_[50][3];
    int waypoint_num_, wp_id_;
    double planning_horizen_, planning_horizen_time_;
    double emergency_time_;
    bool flag_realworld_experiment_;
    bool enable_fail_safe_;
    bool configured_fail_safe_;
    bool report_occupied_start_;

    /* planning data */
    bool have_trigger_, have_target_, have_odom_, have_new_target_, have_recv_pre_agent_;
    bool inflated_escape_attempted_ = false; // reset only by an accepted new target
    FSM_EXEC_STATE exec_state_;
    int continously_called_times_{0};
    /* SwarmDeck: consecutive failed REPLAN_TRAJ ticks, and the back-off they set */
    int replan_failures_{0};
    std::chrono::steady_clock::time_point replan_not_before_{};
    /* SwarmDeck: a goal received before odometry, planned from the first odometry */
    bool have_pending_goal_{false};
    Eigen::Vector3d pending_goal_;
    /* SwarmDeck: the startup handshake on drone_<id>_planning/swarm_trajs */
    bool startup_published_{false};
    double sequential_start_timeout_;
    rclcpp::Time first_odom_time_;
    std::shared_ptr<const traj_utils::msg::MultiBsplines> pending_swarm_trajs_;
    /* SwarmDeck: the newest broadcast per peer received before odometry */
    std::map<size_t, std::shared_ptr<const traj_utils::msg::Bspline>> early_broadcasts_;

    Eigen::Vector3d odom_pos_, odom_vel_, odom_acc_; // odometry state
    Eigen::Quaterniond odom_orient_;

    Eigen::Vector3d init_pt_, start_pt_, start_vel_, start_acc_, start_yaw_; // start state
    Eigen::Vector3d end_pt_, end_vel_;                                       // goal state
    Eigen::Vector3d local_target_pt_, local_target_vel_;                     // local target state
    std::vector<Eigen::Vector3d> wps_;
    int current_wp_;

    bool flag_escape_emergency_{true}; // SwarmDeck: the first emergency stop is published

    // Only the default (planning) callback group owns FSM/optimizer state.
    // The input executor only appends commands and replaces the odometry inbox.
    std::mutex input_mutex_;
    std::deque<traj_utils::msg::PlannerCommand> commands_;
    nav_msgs::msg::Odometry::ConstSharedPtr latest_odom_;
    uint64_t received_sequence_{0}, command_sequence_{0}, goal_sequence_{0};
    bool ordered_commands_{false};
    rclcpp::CallbackGroup::SharedPtr input_group_;
    rclcpp::Subscription<traj_utils::msg::PlannerCommand>::SharedPtr command_sub_;
    rclcpp::Publisher<traj_utils::msg::CommandState>::SharedPtr command_state_pub_;
    void receiveCommand(const traj_utils::msg::PlannerCommand &command);
    void drainInputs();
    bool commandsPending();

    /* ROS utils */
    rclcpp::Node::SharedPtr node_;
    rclcpp::TimerBase::SharedPtr exec_timer_, safety_timer_;

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr waypoint_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<traj_utils::msg::MultiBsplines>::SharedPtr swarm_trajs_sub_;
    rclcpp::Subscription<traj_utils::msg::Bspline>::SharedPtr broadcast_bspline_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr trigger_sub_;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr cancel_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr fsm_state_pub_;
    rclcpp::TimerBase::SharedPtr fsm_state_timer_, idle_broadcast_timer_;

    // rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr replan_pub_;
    // rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr new_pub_;
    rclcpp::Publisher<traj_utils::msg::Bspline>::SharedPtr bspline_pub_;
    rclcpp::Publisher<traj_utils::msg::DataDisp>::SharedPtr data_disp_pub_;
    rclcpp::Publisher<traj_utils::msg::MultiBsplines>::SharedPtr swarm_trajs_pub_;
    rclcpp::Publisher<traj_utils::msg::Bspline>::SharedPtr broadcast_bspline_pub_;

    /* helper functions */
    bool callReboundReplan(bool flag_use_poly_init, bool flag_randomPolyTraj, bool preemptible = true);
    void publishLocalTrajectory();
    bool callEmergencyStop(Eigen::Vector3d stop_pos);                          // front-end and back-end method
    bool planFromGlobalTraj(const int trial_times = 1);
    bool planFromCurrentTraj(const int trial_times = 1, bool preemptible = true);

    /* return value: std::pair< Times of the same state be continuously called, current continuously called state > */
    void changeFSMExecState(FSM_EXEC_STATE new_state, string pos_call);
    std::pair<int, EGOReplanFSM::FSM_EXEC_STATE> timesOfConsecutiveStateCalls();
    void printFSMExecState();
    /* SwarmDeck: the FSM state, for the adapter to follow a goal */
    void publishFSMState();
    void cancelCallback(const std::shared_ptr<const std_msgs::msg::Empty> &msg);

    void readGivenWps();
    void planNextWaypoint(const Eigen::Vector3d next_wp);
    void getLocalTarget();

    /* ROS functions */
    void execFSMCallback();
    void checkCollisionCallback();
    bool escapePoseSafe();
    void waypointCallback(const std::shared_ptr<const geometry_msgs::msg::PoseStamped> &msg);
    void triggerCallback(const std::shared_ptr<const geometry_msgs::msg::PoseStamped> &msg);
    void odometryCallback(const std::shared_ptr<const nav_msgs::msg::Odometry> &msg);
    void swarmTrajsCallback(const std::shared_ptr<const traj_utils::msg::MultiBsplines> &msg);
    void BroadcastBsplineCallback(const std::shared_ptr<const traj_utils::msg::Bspline> &msg);
    bool mergeSwarmTraj(size_t id, const traj_utils::msg::Bspline &traj, const char *source);

    bool checkCollision();
    void publishSwarmTrajs(bool startup_pub);
    void sendSwarmTrajs(const traj_utils::msg::Bspline &bspline, bool startup_pub);
    void publishStartupChain();
    traj_utils::msg::Bspline buildHoverAnnouncement() const;

  public:
    EGOReplanFSM(/* args */)
    {
    }
    ~EGOReplanFSM()
    {
    }

    void init(rclcpp::Node::SharedPtr &node);
    rclcpp::CallbackGroup::SharedPtr inputGroup() const { return input_group_; }
    rclcpp::CallbackGroup::SharedPtr sensingGroup() const { return planner_manager_->grid_map_->callbackGroup(); }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

} // namespace ego_planner

#endif