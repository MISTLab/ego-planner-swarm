
#include <cmath>
#include <ego_planner/ego_replan_fsm.h>
#include <ego_planner/trajectory_publication_check.h>

namespace ego_planner
{
  namespace
  {
    // SwarmDeck: REPLAN_TRAJ retried a failing replan on every 10 ms tick,
    // three optimizations each (drone-r5: a terminal point in an obstacle,
    // about 250 a second until the flown trajectory ran out). Two retries stay
    // immediate; then wait 20 ms, doubling up to 0.5 s.
    std::chrono::milliseconds replanBackoff(int failures)
    {
      if (failures <= 2)
        return std::chrono::milliseconds(0);
      return std::chrono::milliseconds(std::min(500, 20 << std::min(failures - 3, 5)));
    }
  } // namespace

  void EGOReplanFSM::init(rclcpp::Node::SharedPtr &node)
  {
    node_ = node;
    
    current_wp_ = 0;
    exec_state_ = FSM_EXEC_STATE::INIT;
    have_target_ = false;
    have_odom_ = false;
    have_recv_pre_agent_ = false;

    node_->declare_parameter("fsm/ordered_commands", false);
    node_->get_parameter("fsm/ordered_commands", ordered_commands_);
    input_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions input_options;
    input_options.callback_group = input_group_;

    node_->declare_parameter("fsm/flight_type", -1);
    node_->declare_parameter("fsm/thresh_replan_time", -1.0);
    node_->declare_parameter("fsm/thresh_no_replan_meter", -1.0);
    node_->declare_parameter("fsm/planning_horizon", -1.0);
    node_->declare_parameter("fsm/planning_horizen_time", -1.0);
    node_->declare_parameter("fsm/emergency_time", 1.0);
    node_->declare_parameter("fsm/realworld_experiment", false);
    node_->declare_parameter("fsm/fail_safe", true);
    node_->declare_parameter("fsm/report_occupied_start", false);
    node_->declare_parameter("fsm/sequential_start_timeout_s", 10.0);

    node_->get_parameter("fsm/flight_type", target_type_);
    node_->get_parameter("fsm/thresh_replan_time", replan_thresh_);
    node_->get_parameter("fsm/thresh_no_replan_meter", no_replan_thresh_);
    node_->get_parameter("fsm/planning_horizon", planning_horizen_);
    node_->get_parameter("fsm/planning_horizen_time", planning_horizen_time_);
    node_->get_parameter("fsm/emergency_time", emergency_time_);
    node_->get_parameter("fsm/realworld_experiment", flag_realworld_experiment_);
    node_->get_parameter("fsm/fail_safe", enable_fail_safe_);
    configured_fail_safe_ = enable_fail_safe_;
    node_->get_parameter("fsm/report_occupied_start", report_occupied_start_);
    node_->get_parameter("fsm/sequential_start_timeout_s", sequential_start_timeout_);

    have_trigger_ = !flag_realworld_experiment_;

    node_->declare_parameter("fsm/waypoint_num", -1);
    node_->get_parameter("fsm/waypoint_num", waypoint_num_);

    for (int i = 0; i < waypoint_num_; i++)
    {
      node_->declare_parameter("fsm/waypoint" + to_string(i) + "_x", -1.0);
      node_->declare_parameter("fsm/waypoint" + to_string(i) + "_y", -1.0);
      node_->declare_parameter("fsm/waypoint" + to_string(i) + "_z", -1.0);

      node_->get_parameter("fsm/waypoint" + to_string(i) + "_x", waypoints_[i][0]);
      node_->get_parameter("fsm/waypoint" + to_string(i) + "_y", waypoints_[i][1]);
      node_->get_parameter("fsm/waypoint" + to_string(i) + "_z", waypoints_[i][2]);
    }

    /* initialize main modules */
    visualization_.reset(new PlanningVisualization(node_));

    planner_manager_.reset(new EGOPlannerManager);

    planner_manager_->initPlanModules(node_, visualization_);

    planner_manager_->deliverTrajToOptimizer(); // store trajectories
    planner_manager_->setDroneIdtoOpt();

    /* callback*/
    exec_timer_ = node_->create_wall_timer(std::chrono::milliseconds(10),
                                           std::bind(&EGOReplanFSM::execFSMCallback, this));

    safety_timer_ = node_->create_wall_timer(std::chrono::milliseconds(50),
                                             std::bind(&EGOReplanFSM::checkCollisionCallback, this));

    odom_sub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
        "odom_world",
        1,
        [this](const std::shared_ptr<const nav_msgs::msg::Odometry> &msg)
        {
          std::lock_guard<std::mutex> guard(input_mutex_);
          latest_odom_ = msg;
        }, input_options);
    // std::bind(&EGOReplanFSM::odometryCallback, this, std::placeholders::_1));

    if (planner_manager_->pp_.drone_id >= 1)
    {
      string sub_topic_name = string("drone_") + std::to_string(planner_manager_->pp_.drone_id - 1) + string("_planning/swarm_trajs");
      swarm_trajs_sub_ = node_->create_subscription<traj_utils::msg::MultiBsplines>(
          sub_topic_name,
          rclcpp::QoS(1).transient_local(), // SwarmDeck: a late follower still gets it
          [this](const std::shared_ptr<const traj_utils::msg::MultiBsplines> &msg)
          {
            this->swarmTrajsCallback(msg);
          });
    }

    // ros2 中topic名字中不能出现负号，单机id是-1需要处理
    // string pub_topic_name = string("/drone_") + std::to_string(planner_manager_->pp_.drone_id) + string("_planning/swarm_trajs");
    string pub_topic_name;
    if (planner_manager_->pp_.drone_id <= -1)
    {
      RCLCPP_INFO(node_->get_logger(), "single drone:%d", planner_manager_->pp_.drone_id);
      pub_topic_name = string("drone_") + "single" + string("_planning/swarm_trajs");
    }else
    {
      pub_topic_name = string("drone_") + std::to_string(planner_manager_->pp_.drone_id) + string("_planning/swarm_trajs");
    }
    
    swarm_trajs_pub_ = node_->create_publisher<traj_utils::msg::MultiBsplines>(pub_topic_name, rclcpp::QoS(1).transient_local());

    broadcast_bspline_pub_ = node_->create_publisher<traj_utils::msg::Bspline>("planning/broadcast_bspline_from_planner", 10);
    broadcast_bspline_sub_ = node_->create_subscription<traj_utils::msg::Bspline>(
        "planning/broadcast_bspline_to_planner",
        100,
        [this](const std::shared_ptr<const traj_utils::msg::Bspline> &msg)
        {
          this->BroadcastBsplineCallback(msg);
        });

    bspline_pub_ = node_->create_publisher<traj_utils::msg::Bspline>("planning/bspline", 10);
    data_disp_pub_ = node_->create_publisher<traj_utils::msg::DataDisp>("planning/data_display", 100);
    fsm_state_pub_ = node_->create_publisher<std_msgs::msg::String>(
        "planning/fsm_state", rclcpp::QoS(1).transient_local());
    command_state_pub_ = node_->create_publisher<traj_utils::msg::CommandState>(
        "planning/command_state", rclcpp::QoS(100).transient_local());
    // State publication belongs to the planning owner: identity and state agree.
    fsm_state_timer_ = node_->create_wall_timer(std::chrono::seconds(1),
                                                std::bind(&EGOReplanFSM::publishFSMState, this));

    idle_broadcast_timer_ = node_->create_wall_timer(std::chrono::seconds(1), [this]() {
      if (!have_odom_ || planner_manager_->pp_.drone_id < 0)
        return;
      const auto &local = planner_manager_->local_data_;
      const bool active = have_target_ && local.start_time_.seconds() > 0.0 &&
          (rclcpp::Clock().now() - local.start_time_).seconds() < local.duration_;
      if (!active)
        broadcast_bspline_pub_->publish(buildHoverAnnouncement());
    });

    if (target_type_ == TARGET_TYPE::MANUAL_TARGET)
    {
      if (ordered_commands_)
      {
        command_sub_ = node_->create_subscription<traj_utils::msg::PlannerCommand>(
            "command", rclcpp::QoS(rclcpp::KeepAll()).reliable(),
            [this](traj_utils::msg::PlannerCommand::ConstSharedPtr msg) { receiveCommand(*msg); }, input_options);
      }
      else
      {
        // Legacy messages have no shared publication sequence. Compatibility
        // mode preserves receipt order only; never mix it with ordered input.
        waypoint_sub_ = node_->create_subscription<geometry_msgs::msg::PoseStamped>(
            "goal", rclcpp::QoS(rclcpp::KeepAll()).reliable(),
            [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) {
              traj_utils::msg::PlannerCommand command;
              command.command = traj_utils::msg::PlannerCommand::GOAL;
              command.goal = *msg;
              receiveCommand(command);
            }, input_options);
        cancel_sub_ = node_->create_subscription<std_msgs::msg::Empty>(
            "cancel", rclcpp::QoS(rclcpp::KeepAll()).reliable(),
            [this](std_msgs::msg::Empty::ConstSharedPtr) {
              traj_utils::msg::PlannerCommand command;
              command.command = traj_utils::msg::PlannerCommand::CANCEL;
              receiveCommand(command);
            }, input_options);
      }
    }
    else if (target_type_ == TARGET_TYPE::PRESET_TARGET)
    {
      trigger_sub_ = node_->create_subscription<geometry_msgs::msg::PoseStamped>(
          "traj_start_trigger",
          1,
          [this](const std::shared_ptr<const geometry_msgs::msg::PoseStamped> &msg)
          {
            this->triggerCallback(msg);
          });

      RCLCPP_INFO(node_->get_logger(), "Wait for 1 second.");
      int count = 0;
      while (rclcpp::ok() && count++ < 1000)
      {
        rclcpp::spin_some(node_);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

      RCLCPP_WARN(node_->get_logger(), "Waiting for trigger from [n3ctrl] from RC");

      while (rclcpp::ok() && (!have_odom_ || !have_trigger_))
      {
        rclcpp::spin_some(node_);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

      readGivenWps();
    }
    else
      cout << "Wrong target_type_ value! target_type_=" << target_type_ << endl;
  }

  void EGOReplanFSM::receiveCommand(const traj_utils::msg::PlannerCommand &command)
  {
    using Command = traj_utils::msg::PlannerCommand;
    if (command.command != Command::GOAL && command.command != Command::CANCEL)
      return;
    const auto &p = command.goal.pose.position;
    if (command.command == Command::GOAL &&
        (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)))
      return;
    std::lock_guard<std::mutex> guard(input_mutex_);
    if (ordered_commands_ && command.sequence <= received_sequence_)
      return;
    received_sequence_ = ordered_commands_ ? command.sequence : received_sequence_ + 1;
    commands_.push_back(command);
  }

  bool EGOReplanFSM::commandsPending()
  {
    std::lock_guard<std::mutex> guard(input_mutex_);
    return !commands_.empty();
  }

  void EGOReplanFSM::drainInputs()
  {
    std::deque<traj_utils::msg::PlannerCommand> commands;
    nav_msgs::msg::Odometry::ConstSharedPtr odom;
    {
      std::lock_guard<std::mutex> guard(input_mutex_);
      commands.swap(commands_);
      odom.swap(latest_odom_);
    }
    if (odom) odometryCallback(odom);
    for (const auto &command : commands)
    {
      command_sequence_ = command.sequence;
      if (command.command == traj_utils::msg::PlannerCommand::GOAL)
      {
        goal_sequence_ = command.sequence;
        waypointCallback(std::make_shared<geometry_msgs::msg::PoseStamped>(command.goal));
      }
      else
        cancelCallback(nullptr);
      // Emit even if a new goal leaves the FSM in the same state.
      publishFSMState();
    }
  }

  void EGOReplanFSM::readGivenWps()

  {
    if (waypoint_num_ <= 0)
    {
      RCLCPP_ERROR(node_->get_logger(), "Wrong waypoint_num_ = %d", waypoint_num_);
      return;
    }

    wps_.resize(waypoint_num_);
    for (int i = 0; i < waypoint_num_; i++)
    {
      wps_[i](0) = waypoints_[i][0];
      wps_[i](1) = waypoints_[i][1];
      wps_[i](2) = waypoints_[i][2];
    }

    // 用 visualization_->displayGoalPoint() 方法对waypoint进行可视化
    for (size_t i = 0; i < (size_t)waypoint_num_; i++)
    {
      visualization_->displayGoalPoint(wps_[i], Eigen::Vector4d(0, 0.5, 0.5, 1), 0.3, i);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // plan first global waypoint
    wp_id_ = 0;
    planNextWaypoint(wps_[wp_id_]);
  }

  void EGOReplanFSM::planNextWaypoint(const Eigen::Vector3d next_wp)
  {
    if (planner_manager_->grid_map_->getOdomDepthTimeout())
    {
      // Acknowledge the newest goal, but never leave the stop to plan on a
      // stale map. Both legacy and ordered goals resume after fresh sensing.
      pending_goal_ = next_wp;
      have_pending_goal_ = true;
      enable_fail_safe_ = false;
      if (exec_state_ != EMERGENCY_STOP || flag_escape_emergency_)
        callEmergencyStop(odom_pos_);
      flag_escape_emergency_ = false;
      if (exec_state_ != EMERGENCY_STOP)
        changeFSMExecState(EMERGENCY_STOP, "SAFETY");
      return;
    }
    have_pending_goal_ = false;
    bool success = false;
    success = planner_manager_->planGlobalTraj(odom_pos_, odom_vel_, Eigen::Vector3d::Zero(), next_wp, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    if (success)
    {
      end_pt_ = next_wp;

      constexpr double step_size_t = 0.1;
      int i_end = floor(planner_manager_->global_data_.global_duration_ / step_size_t);
      vector<Eigen::Vector3d> gloabl_traj(i_end);
      for (int i = 0; i < i_end; i++)
      {
        gloabl_traj[i] = planner_manager_->global_data_.global_traj_.evaluate(i * step_size_t);
      }

      end_vel_.setZero();
      have_target_ = true;
      have_new_target_ = true;
      inflated_escape_attempted_ = false;

      /*** FSM state change (SwarmDeck: never block the executor) ***/
      enable_fail_safe_ = configured_fail_safe_;
      if (exec_state_ == INIT)
      {
        // WAIT_TARGET takes the target once odometry has arrived.
      }
      else if (!startup_published_)
        changeFSMExecState(SEQUENTIAL_START, "TRIG");
      else if (exec_state_ == EXEC_TRAJ)
        changeFSMExecState(REPLAN_TRAJ, "TRIG");
      else
        changeFSMExecState(GEN_NEW_TRAJ, "TRIG");

      visualization_->displayGlobalPathList(gloabl_traj, 0.1, 0);
    }
    else
    {
      RCLCPP_ERROR(node_->get_logger(), "Unable to generate global trajectory!");
    }
  }

  void EGOReplanFSM::triggerCallback(const std::shared_ptr<const geometry_msgs::msg::PoseStamped> &msg)
  {
    have_trigger_ = true;
    cout << "Triggered!" << endl;
    init_pt_ = odom_pos_;
  }

  void EGOReplanFSM::waypointCallback(const std::shared_ptr<const geometry_msgs::msg::PoseStamped> &msg)
  {
    const auto &p = msg->pose.position;
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
      return;

    cout << "Triggered!" << endl;

    have_trigger_ = true;
    if (!have_odom_)
    {
      // The global trajectory starts at the drone: plan it from the first odometry.
      pending_goal_ = Eigen::Vector3d(p.x, p.y, p.z);
      have_pending_goal_ = true;
      RCLCPP_INFO(node_->get_logger(), "No odometry yet: the goal waits for it.");
      return;
    }

    init_pt_ = odom_pos_;
    planNextWaypoint(Eigen::Vector3d(p.x, p.y, p.z));
  }

  void EGOReplanFSM::odometryCallback(const std::shared_ptr<const nav_msgs::msg::Odometry> &msg)
  {
    odom_pos_(0) = msg->pose.pose.position.x;
    odom_pos_(1) = msg->pose.pose.position.y;
    odom_pos_(2) = msg->pose.pose.position.z;

    odom_vel_(0) = msg->twist.twist.linear.x;
    odom_vel_(1) = msg->twist.twist.linear.y;
    odom_vel_(2) = msg->twist.twist.linear.z;

    // odom_acc_ = estimateAcc( msg );

    odom_orient_.w() = msg->pose.pose.orientation.w;
    odom_orient_.x() = msg->pose.pose.orientation.x;
    odom_orient_.y() = msg->pose.pose.orientation.y;
    odom_orient_.z() = msg->pose.pose.orientation.z;

    if (!have_odom_)
      first_odom_time_ = rclcpp::Clock().now();
    have_odom_ = true;

    if (pending_swarm_trajs_)
    {
      auto startup = pending_swarm_trajs_;
      pending_swarm_trajs_.reset();
      swarmTrajsCallback(startup);
    }

    if (!early_broadcasts_.empty())
    {
      // Their freshness was checked on arrival and would fail now, since they
      // waited for this odometry; the distance test, which needed our
      // position, runs in mergeSwarmTraj like for any other trajectory.
      for (const auto &kept : early_broadcasts_)
        if (mergeSwarmTraj(kept.first, *kept.second, "Early"))
          RCLCPP_INFO(node_->get_logger(), "Kept the trajectory of drone %d received before odometry.", (int)kept.first);
      early_broadcasts_.clear();
    }

    if (have_pending_goal_ && !planner_manager_->grid_map_->getOdomDepthTimeout())
    {
      have_pending_goal_ = false;
      init_pt_ = odom_pos_;
      planNextWaypoint(pending_goal_);
    }
  }

  void EGOReplanFSM::BroadcastBsplineCallback(const std::shared_ptr<const traj_utils::msg::Bspline> &msg)
  {
    size_t id = msg->drone_id;
    if ((int)id == planner_manager_->pp_.drone_id)
      return;

    // if (abs((ros::Time::now() - msg->start_time).toSec()) > 0.25)
    rclcpp::Clock clock(RCL_SYSTEM_TIME);  // 确保使用当前节点的时间源
    auto msg_time = rclcpp::Time(msg->start_time, clock.get_clock_type());
    // RCLCPP_INFO(node_->get_logger(), "Clock type: %d", rclcpp::Clock().now().get_clock_type());
    // RCLCPP_INFO(node_->get_logger(), "Start time clock type: %d", rclcpp::Time(msg->start_time).get_clock_type());
    // RCLCPP_INFO(node_->get_logger(), "msg_time: %d", msg_time.get_clock_type());
    if (abs((rclcpp::Clock().now() - msg_time).seconds()) > 0.25)
    {
      // ROS_ERROR("Time difference is too large! Local - Remote Agent %d = %fs", msg->drone_id, (ros::Time::now() - msg->start_time).toSec());
      RCLCPP_ERROR(node_->get_logger(), "Time difference is too large! Local - Remote Agent %d = %fs",
                   msg->drone_id, (rclcpp::Clock().now() - msg_time).seconds());
      return;
    }

    // SwarmDeck: the distance test below needs our position, which is unset
    // before the first odometry. Keep the newest broadcast of each peer until
    // then (an idle peer never sends another); the first odometry merges it.
    if (!have_odom_)
    {
      if (msg->order == 3 && msg->pos_pts.size() >= 3 && msg->knots.size() >= 2)
      {
        std::shared_ptr<const traj_utils::msg::Bspline> &kept = early_broadcasts_[id];
        if (!kept || rclcpp::Time(kept->start_time).seconds() < rclcpp::Time(msg->start_time).seconds())
          kept = msg;
      }
      return;
    }

    // 路径缓冲区初始化
    if (planner_manager_->swarm_trajs_buf_.size() <= id)
    {
      for (size_t i = planner_manager_->swarm_trajs_buf_.size(); i <= id; i++)
      {
        OneTrajDataOfSwarm blank;
        blank.drone_id = -1;
        planner_manager_->swarm_trajs_buf_.push_back(blank);
      }
    }

    /* Test distance to the agent */
    Eigen::Vector3d cp0(msg->pos_pts[0].x, msg->pos_pts[0].y, msg->pos_pts[0].z);
    Eigen::Vector3d cp1(msg->pos_pts[1].x, msg->pos_pts[1].y, msg->pos_pts[1].z);
    Eigen::Vector3d cp2(msg->pos_pts[2].x, msg->pos_pts[2].y, msg->pos_pts[2].z);
    Eigen::Vector3d swarm_start_pt = (cp0 + 4 * cp1 + cp2) / 6;
    if ((swarm_start_pt - odom_pos_).norm() > planning_horizen_ * 4.0f / 3.0f)
    {
      planner_manager_->swarm_trajs_buf_[id].drone_id = -1;
      return; // if the current drone is too far to the received agent.
    }

    /* Store data */
    Eigen::MatrixXd pos_pts(3, msg->pos_pts.size());
    Eigen::VectorXd knots(msg->knots.size());
    for (size_t j = 0; j < msg->knots.size(); ++j)
    {
      knots(j) = msg->knots[j];
    }
    for (size_t j = 0; j < msg->pos_pts.size(); ++j)
    {
      pos_pts(0, j) = msg->pos_pts[j].x;
      pos_pts(1, j) = msg->pos_pts[j].y;
      pos_pts(2, j) = msg->pos_pts[j].z;
    }

    planner_manager_->swarm_trajs_buf_[id].drone_id = id;

    // 计算路径持续时间
    if (msg->order % 2)
    {
      double cutback = (double)msg->order / 2 + 1.5;
      planner_manager_->swarm_trajs_buf_[id].duration_ = msg->knots[msg->knots.size() - ceil(cutback)];
    }
    else
    {
      double cutback = (double)msg->order / 2 + 1.5;
      planner_manager_->swarm_trajs_buf_[id].duration_ = (msg->knots[msg->knots.size() - floor(cutback)] + msg->knots[msg->knots.size() - ceil(cutback)]) / 2;
    }

    // 生成bspline并存储
    UniformBspline pos_traj(pos_pts, msg->order, msg->knots[1] - msg->knots[0]);
    pos_traj.setKnot(knots);
    planner_manager_->swarm_trajs_buf_[id].position_traj_ = pos_traj;

    planner_manager_->swarm_trajs_buf_[id].start_pos_ = planner_manager_->swarm_trajs_buf_[id].position_traj_.evaluateDeBoorT(0);

    planner_manager_->swarm_trajs_buf_[id].start_time_ = msg->start_time;

    /* Check Collision (SwarmDeck: only a trajectory flown toward a target is replanned) */
    if (have_target_ && exec_state_ == EXEC_TRAJ && planner_manager_->checkCollision(id))
    {
      changeFSMExecState(REPLAN_TRAJ, "TRAJ_CHECK");
    }
  }

  void EGOReplanFSM::swarmTrajsCallback(const std::shared_ptr<const traj_utils::msg::MultiBsplines> &msg)
  {
    if (startup_published_) // SwarmDeck: the chain is a startup handshake only
      return;

    multi_bspline_msgs_buf_.traj.clear();
    multi_bspline_msgs_buf_ = *msg;

    if (!have_odom_)
    {
      RCLCPP_INFO(node_->get_logger(), "swarmTrajsCallback(): no odom yet, kept for the first odometry.");
      pending_swarm_trajs_ = msg;
      return;
    }

    if ((int)msg->traj.size() != msg->drone_id_from + 1) // drone_id must start from 0
    {
      RCLCPP_ERROR(node_->get_logger(), "Wrong trajectory size!msg->traj.size()=%d, msg->drone_id_from+1=%d", (int)msg->traj.size(), msg->drone_id_from + 1);
      return;
    }

    // Step 1. merge the startup trajectories (SwarmDeck: into the live buffer,
    // which is never cleared; an entry only fills a missing or older one, since
    // a retained startup snapshot can be older than the peer's broadcasts).
    for (size_t i = 0; i < msg->traj.size(); i++)
    {
      const traj_utils::msg::Bspline &traj = msg->traj[i];

      // only support B-spline order equals 3; an absent drone leaves an empty entry.
      if ((int)i == planner_manager_->pp_.drone_id || traj.order != 3 || traj.pos_pts.size() < 3 || traj.knots.size() < 2)
        continue;

      mergeSwarmTraj(i, traj, "Startup");
    }

    have_recv_pre_agent_ = true;
  }

  bool EGOReplanFSM::mergeSwarmTraj(size_t id, const traj_utils::msg::Bspline &traj, const char *source)
  {
    // SwarmDeck: a trajectory that is not live (a startup snapshot, or a
    // broadcast kept from before odometry) only fills a missing entry or
    // replaces an older one, and never clears the buffer.
    SwarmTrajData &buf = planner_manager_->swarm_trajs_buf_;
    if (buf.size() > id && buf[id].drone_id == (int)id &&
        buf[id].start_time_.seconds() >= rclcpp::Time(traj.start_time).seconds())
    {
      RCLCPP_INFO(node_->get_logger(), "%s trajectory of drone %d is older than its live one: kept the live one.", source, (int)id);
      return false;
    }

    Eigen::Vector3d cp0(traj.pos_pts[0].x, traj.pos_pts[0].y, traj.pos_pts[0].z);
    Eigen::Vector3d cp1(traj.pos_pts[1].x, traj.pos_pts[1].y, traj.pos_pts[1].z);
    Eigen::Vector3d cp2(traj.pos_pts[2].x, traj.pos_pts[2].y, traj.pos_pts[2].z);
    Eigen::Vector3d swarm_start_pt = (cp0 + 4 * cp1 + cp2) / 6;
    if ((swarm_start_pt - odom_pos_).norm() > planning_horizen_ * 4.0f / 3.0f)
      return false;

    for (size_t j = buf.size(); j <= id; j++)
    {
      OneTrajDataOfSwarm blank;
      blank.drone_id = -1;
      buf.push_back(blank);
    }

    // 存储路径控制点和节点
    Eigen::MatrixXd pos_pts(3, traj.pos_pts.size());
    Eigen::VectorXd knots(traj.knots.size());
    for (size_t j = 0; j < traj.knots.size(); ++j)
    {
      knots(j) = traj.knots[j];
    }
    for (size_t j = 0; j < traj.pos_pts.size(); ++j)
    {
      pos_pts(0, j) = traj.pos_pts[j].x;
      pos_pts(1, j) = traj.pos_pts[j].y;
      pos_pts(2, j) = traj.pos_pts[j].z;
    }

    buf[id].drone_id = id;

    // 计算路径持续时间
    if (traj.order % 2)
    {
      double cutback = (double)traj.order / 2 + 1.5;
      buf[id].duration_ = traj.knots[traj.knots.size() - ceil(cutback)];
    }
    else
    {
      double cutback = (double)traj.order / 2 + 1.5;
      buf[id].duration_ = (traj.knots[traj.knots.size() - floor(cutback)] + traj.knots[traj.knots.size() - ceil(cutback)]) / 2;
    }

    UniformBspline pos_traj(pos_pts, traj.order, traj.knots[1] - traj.knots[0]);
    pos_traj.setKnot(knots);
    buf[id].position_traj_ = pos_traj;

    buf[id].start_pos_ = buf[id].position_traj_.evaluateDeBoorT(0);

    buf[id].start_time_ = traj.start_time;
    return true;
  }

  void EGOReplanFSM::changeFSMExecState(FSM_EXEC_STATE new_state, string pos_call)
  {

    if (new_state == exec_state_)
      continously_called_times_++;
    else
    {
      continously_called_times_ = 1;
      // Any other state (a success, a new goal, a cancel) ends the back-off.
      replan_failures_ = 0;
      replan_not_before_ = std::chrono::steady_clock::time_point();
    }

    static string state_str[9] = {"INIT", "WAIT_TARGET", "GEN_NEW_TRAJ", "REPLAN_TRAJ", "EXEC_TRAJ", "EMERGENCY_STOP", "SEQUENTIAL_START", "OCCUPIED_START", "INFLATED_ESCAPE"};
    int pre_s = int(exec_state_);
    exec_state_ = new_state;
    cout << "[" + pos_call + "]: from " + state_str[pre_s] + " to " + state_str[int(new_state)] << endl;
    if (int(new_state) != pre_s)
      publishFSMState();
  }

  std::pair<int, EGOReplanFSM::FSM_EXEC_STATE> EGOReplanFSM::timesOfConsecutiveStateCalls()
  {
    return std::pair<int, FSM_EXEC_STATE>(continously_called_times_, exec_state_);
  }

  void EGOReplanFSM::printFSMExecState()
  {
    static string state_str[9] = {"INIT", "WAIT_TARGET", "GEN_NEW_TRAJ", "REPLAN_TRAJ", "EXEC_TRAJ", "EMERGENCY_STOP", "SEQUENTIAL_START", "OCCUPIED_START", "INFLATED_ESCAPE"};

    cout << "[FSM]: state: " + state_str[int(exec_state_)] << endl;
  }

  void EGOReplanFSM::publishFSMState()
  {
    static const string state_str[9] = {"INIT", "WAIT_TARGET", "GEN_NEW_TRAJ", "REPLAN_TRAJ", "EXEC_TRAJ", "EMERGENCY_STOP", "SEQUENTIAL_START", "OCCUPIED_START", "EXEC_TRAJ"};
    std_msgs::msg::String msg;
    msg.data = state_str[int(exec_state_)];
    fsm_state_pub_->publish(msg);
    traj_utils::msg::CommandState attributed;
    attributed.command_sequence = command_sequence_;
    attributed.goal_sequence = goal_sequence_;
    attributed.state = msg.data;
    command_state_pub_->publish(attributed);
  }

  void EGOReplanFSM::cancelCallback(const std::shared_ptr<const std_msgs::msg::Empty> &)
  {
    have_target_ = false;
    have_pending_goal_ = false;
    if (exec_state_ == GEN_NEW_TRAJ || exec_state_ == REPLAN_TRAJ || exec_state_ == EXEC_TRAJ || exec_state_ == EMERGENCY_STOP || exec_state_ == INFLATED_ESCAPE)
    {
      // A real stop: traj_server would otherwise fly the rest of the last trajectory.
      callEmergencyStop(odom_pos_);
      publishSwarmTrajs(false);
    }
    if (exec_state_ != INIT)
      changeFSMExecState(WAIT_TARGET, "CANCEL");
  }

  void EGOReplanFSM::publishStartupChain()
  {
    // SwarmDeck: each drone tells its follower it is up as soon as it has
    // odometry and has heard from its predecessor (or waited
    // fsm/sequential_start_timeout_s for it), whether or not it has a goal.
    if (startup_published_ || !have_odom_)
      return;

    const int drone_id = planner_manager_->pp_.drone_id;
    if (drone_id >= 1 && !have_recv_pre_agent_)
    {
      const double waited = (rclcpp::Clock().now() - first_odom_time_).seconds();
      if (waited < sequential_start_timeout_)
        return;
      RCLCPP_WARN(node_->get_logger(), "No startup trajectory from drone %d after %.1f s: starting without it.", drone_id - 1, waited);
    }

    if (drone_id >= 0)
    {
      sendSwarmTrajs(buildHoverAnnouncement(), true);
    }
    startup_published_ = true;
  }

  traj_utils::msg::Bspline EGOReplanFSM::buildHoverAnnouncement() const
  {
    // EmergencyStop's shape, built independently: it must never become a
    // flown local trajectory or reach traj_server through planning/bspline.
    traj_utils::msg::Bspline hover;
    hover.order = 3;
    hover.start_time = rclcpp::Clock().now();
    hover.drone_id = planner_manager_->pp_.drone_id;
    hover.traj_id = 0;
    geometry_msgs::msg::Point pt;
    pt.x = odom_pos_(0);
    pt.y = odom_pos_(1);
    pt.z = odom_pos_(2);
    hover.pos_pts.assign(6, pt);
    for (int i = 0; i < (int)hover.pos_pts.size() + hover.order + 1; ++i)
      hover.knots.push_back(double(i - hover.order));
    return hover;
  }

  void EGOReplanFSM::execFSMCallback()
  {
    exec_timer_->cancel(); // To avoid blockage
    drainInputs();
    if (have_odom_ && have_pending_goal_ && !planner_manager_->grid_map_->getOdomDepthTimeout())
    {
      init_pt_ = odom_pos_;
      planNextWaypoint(pending_goal_);
    }

    static int fsm_num = 0;
    fsm_num++;
    if (fsm_num == 100)
    {
      printFSMExecState();
      if (!have_odom_)
        cout << "no odom." << endl;
      if (!have_target_)
        cout << "wait for goal or trigger." << endl;
      fsm_num = 0;
    }

    publishStartupChain();

    // The optimizer fixes the starting control points: it cannot manufacture
    // a free prefix inside an obstacle. Yield to an external, bounded retreat;
    // never move the spline's start to a fictitious nearest free cell.
    // Tracking replans start on our existing spline, not at the drifted pose.
    // Odometry-start plans need a free start. Before yielding to external
    // retreat, try a bounded, certified inflation or vertical band correction.
    const double pose_band_tolerance =
        (exec_state_ == EXEC_TRAJ || exec_state_ == REPLAN_TRAJ) ? 0.1 : 0.0;
    if (report_occupied_start_ && have_odom_ && have_target_ && exec_state_ != INFLATED_ESCAPE &&
        planner_manager_->grid_map_->getInflateOccupancy(odom_pos_, pose_band_tolerance) != 0)
    {
      auto guard = planner_manager_->grid_map_->lock();
      planner_manager_->grid_map_->logOccupiedStart(odom_pos_);
      RCLCPP_WARN(node_->get_logger(), "OCCUPIED_START admission speed=%.4f attempted=%d pending_command=%d",
                  odom_vel_.norm(), inflated_escape_attempted_, commandsPending());
      if (!inflated_escape_attempted_ && odom_vel_.norm() < 0.1 &&
          !planner_manager_->grid_map_->getOdomDepthTimeout() &&
          planner_manager_->inflatedStartEscape(odom_pos_, end_pt_) && !commandsPending())
      {
        publishLocalTrajectory();
        publishSwarmTrajs(false);
        inflated_escape_attempted_ = true;
        changeFSMExecState(INFLATED_ESCAPE, "INFLATED_ESCAPE");
        goto force_return;
      }
      callEmergencyStop(odom_pos_);
      publishSwarmTrajs(false);
      have_target_ = false;
      have_pending_goal_ = false;
      changeFSMExecState(OCCUPIED_START, "OCCUPIED_START");
    }

    switch (exec_state_)
    {
    case INIT:
    {
      if (!have_odom_)
      {
        goto force_return;
      }
      changeFSMExecState(WAIT_TARGET, "FSM");
      break;
    }

    case INFLATED_ESCAPE:
    {
      auto &info = planner_manager_->local_data_;
      if ((rclcpp::Clock().now()-info.start_time_).seconds() >= info.duration_)
      {
        if (planner_manager_->grid_map_->getInflateOccupancy(odom_pos_) == 0)
          changeFSMExecState(GEN_NEW_TRAJ, "ESCAPED");
        else
        {
          callEmergencyStop(odom_pos_);
          have_target_ = false;
          have_pending_goal_ = false;
          changeFSMExecState(OCCUPIED_START, "ESCAPE_NOT_TRACKED");
        }
      }
      break;
    }

    case OCCUPIED_START:
      break; // cancellation or a new goal after external recovery only

    case WAIT_TARGET:
    {
      if (!have_target_ || !have_trigger_)
        goto force_return;
      else
      {
        changeFSMExecState(SEQUENTIAL_START, "FSM");
      }
      break;
    }

    case SEQUENTIAL_START: // for swarm
    {
      // SwarmDeck: wait for the startup handshake (publishStartupChain), then
      // plan the first trajectory as any other.
      if (startup_published_)
        changeFSMExecState(GEN_NEW_TRAJ, "FSM");
      break;
    }

    case GEN_NEW_TRAJ:
    {
      if (!have_target_) // SwarmDeck: never plan toward a dropped target
      {
        changeFSMExecState(WAIT_TARGET, "FSM");
        break;
      }

      bool success = planFromGlobalTraj(10); // zx-todo
      if (success)
      {
        changeFSMExecState(EXEC_TRAJ, "FSM");
        flag_escape_emergency_ = true;
        publishSwarmTrajs(false);
      }
      else
      {
        changeFSMExecState(GEN_NEW_TRAJ, "FSM");
      }
      break;
    }

    case REPLAN_TRAJ:
    {
      if (!have_target_) // SwarmDeck: never plan toward a dropped target
      {
        changeFSMExecState(WAIT_TARGET, "FSM");
        break;
      }

      if (std::chrono::steady_clock::now() < replan_not_before_)
        break; // backing off: replanBackoff

      if (planFromCurrentTraj(1))
      {
        changeFSMExecState(EXEC_TRAJ, "FSM");
        publishSwarmTrajs(false);
      }
      else
      {
        replan_failures_++;
        replan_not_before_ = std::chrono::steady_clock::now() + replanBackoff(replan_failures_);
        changeFSMExecState(REPLAN_TRAJ, "FSM");
      }

      break;
    }

    case EXEC_TRAJ:
    {
      /* determine if need to replan */
      LocalTrajData *info = &planner_manager_->local_data_;
      rclcpp::Time time_now = rclcpp::Clock().now();
      double t_cur = (time_now - info->start_time_).seconds();
      t_cur = std::min(info->duration_, t_cur);

      Eigen::Vector3d pos = info->position_traj_.evaluateDeBoorT(t_cur);

      /* && (end_pt_ - pos).norm() < 0.5 */
      if ((target_type_ == TARGET_TYPE::PRESET_TARGET) &&
          (wp_id_ < waypoint_num_ - 1) &&
          (end_pt_ - pos).norm() < no_replan_thresh_)
      {
        wp_id_++;
        planNextWaypoint(wps_[wp_id_]);
      }
      else if ((local_target_pt_ - end_pt_).norm() < 1e-3) // close to the global target
      {
        if (t_cur > info->duration_ - 1e-2)
        {
          have_target_ = false;
          have_trigger_ = false;

          if (target_type_ == TARGET_TYPE::PRESET_TARGET)
          {
            wp_id_ = 0;
            planNextWaypoint(wps_[wp_id_]);
          }

          changeFSMExecState(WAIT_TARGET, "FSM");
          goto force_return;
        }
        else if ((end_pt_ - pos).norm() > no_replan_thresh_ && t_cur > replan_thresh_)
        {
          changeFSMExecState(REPLAN_TRAJ, "FSM");
        }
      }
      else if (t_cur > replan_thresh_)
      {
        changeFSMExecState(REPLAN_TRAJ, "FSM");
      }

      break;
    }

    case EMERGENCY_STOP:
    {

      if (flag_escape_emergency_) // Avoiding repeated calls
      {
        callEmergencyStop(odom_pos_);
      }
      else
      {
        if (enable_fail_safe_ && odom_vel_.norm() < 0.1)
          changeFSMExecState(GEN_NEW_TRAJ, "FSM");
      }

      flag_escape_emergency_ = false;
      break;
    }
    }

    data_disp_.header.stamp = rclcpp::Clock().now();
    data_disp_pub_->publish(data_disp_);

  force_return:;
    // exec_timer_.start();
    if (exec_timer_ && exec_timer_->is_canceled())
    {
      // 取消状态下无需重新创建，可以复用现有计时器
      exec_timer_->reset();
    }
  }

  bool EGOReplanFSM::planFromGlobalTraj(const int trial_times /*=1*/) // zx-todo
  {
    start_pt_ = odom_pos_;
    start_vel_ = odom_vel_;
    start_acc_.setZero();

    bool flag_random_poly_init;
    if (timesOfConsecutiveStateCalls().first == 1)
      flag_random_poly_init = false;
    else
      flag_random_poly_init = true;

    for (int i = 0; i < trial_times && !commandsPending(); i++)
    {
      if (callReboundReplan(true, flag_random_poly_init))
      {
        return true;
      }
    }
    return false;
  }

  bool EGOReplanFSM::planFromCurrentTraj(const int trial_times /*=1*/, bool preemptible /*=true*/)
  {

    LocalTrajData *info = &planner_manager_->local_data_;
    // ros::Time time_now = ros::Time::now();
    auto time_now = rclcpp::Clock().now();
    // double t_cur = (time_now - info->start_time_).toSec();
    double t_cur = (time_now - info->start_time_).seconds();

    start_pt_ = info->position_traj_.evaluateDeBoorT(t_cur);
    start_vel_ = info->velocity_traj_.evaluateDeBoorT(t_cur);
    start_acc_ = info->acceleration_traj_.evaluateDeBoorT(t_cur);

    bool success = callReboundReplan(false, false, preemptible);

    if (!success)
    {
      success = callReboundReplan(true, false, preemptible);
      if (!success)
      {
        for (int i = 0; i < trial_times; i++)
        {
          success = callReboundReplan(true, true, preemptible);
          if (success)
            break;
        }
        if (!success)
        {
          return false;
        }
      }
    }

    return true;
  }

  void EGOReplanFSM::checkCollisionCallback()
  {
    drainInputs();

    LocalTrajData *info = &planner_manager_->local_data_;
    auto map = planner_manager_->grid_map_;
    
    if (!have_target_ || exec_state_ == INIT || exec_state_ == WAIT_TARGET || exec_state_ == SEQUENTIAL_START || info->start_time_.seconds() < 1e-5)
      return;

    /* ---------- check lost of depth ---------- */
    if (map->getOdomDepthTimeout())
    {
      RCLCPP_ERROR(node_->get_logger(), "Depth Lost! EMERGENCY_STOP");

      enable_fail_safe_ = false;
      if (exec_state_ != EMERGENCY_STOP || flag_escape_emergency_)
        callEmergencyStop(odom_pos_);
      flag_escape_emergency_ = false;
      if (exec_state_ != EMERGENCY_STOP)
        changeFSMExecState(EMERGENCY_STOP, "SAFETY");
      return; // local_data_ now holds the stop, not the trajectory below
    }

    double escape_low, escape_high;
    Eigen::Vector3d escape_end = odom_pos_;
    if (exec_state_ == INFLATED_ESCAPE)
      escape_end = info->position_traj_.evaluateDeBoorT(info->duration_);
    double start_low, start_high;
    const bool band_correction = exec_state_ == INFLATED_ESCAPE &&
        map->flightBandLimits(info->start_pos_, start_low, start_high) &&
        (info->start_pos_.z() < start_low || info->start_pos_.z() > start_high);
    const bool escape_pose_blocked = exec_state_ == INFLATED_ESCAPE &&
        (band_correction
         ? !map->bandEscapeTrackingSafe(info->start_pos_, escape_end, odom_pos_)
         : (map->getOccupancy(odom_pos_) != 0 ||
            (map->flightBandLimits(odom_pos_, escape_low, escape_high) &&
             (odom_pos_.z() < escape_low || odom_pos_.z() > escape_high)) ||
            !map->escapeSegmentSafe(info->start_pos_, escape_end)));
    if (escape_pose_blocked)
    {
      callEmergencyStop(odom_pos_);
      have_target_ = false;
      have_pending_goal_ = false;
      changeFSMExecState(OCCUPIED_START, "ESCAPE_BLOCKED");
      return;
    }

    /* ---------- check trajectory ---------- */
    constexpr double time_step = 0.01;
    // double t_cur = (ros::Time::now() - info->start_time_).toSec();
    double t_cur = (rclcpp::Clock().now() - info->start_time_).seconds();

    Eigen::Vector3d p_cur = info->position_traj_.evaluateDeBoorT(t_cur);
    const double CLEARANCE = 1.0 * planner_manager_->getSwarmClearance();
    // double t_cur_global = ros::Time::now().toSec();
    double t_cur_global = rclcpp::Clock().now().seconds();

    double t_2_3 = info->duration_ * 2 / 3;
    for (double t = t_cur; t < info->duration_; t += time_step)
    {
      if (exec_state_ != INFLATED_ESCAPE && t_cur < t_2_3 && t >= t_2_3) // If t_cur < t_2_3, only the first 2/3 partition of the trajectory is considered valid and will get checked.
        break;

      bool occ = false;
      occ |= exec_state_ == INFLATED_ESCAPE
          ? map->getOccupancy(info->position_traj_.evaluateDeBoorT(t)) != 0
          : map->getInflateOccupancy(info->position_traj_.evaluateDeBoorT(t)) != 0;

      for (size_t id = 0; id < planner_manager_->swarm_trajs_buf_.size(); id++)
      {
        if ((planner_manager_->swarm_trajs_buf_.at(id).drone_id != (int)id) || (planner_manager_->swarm_trajs_buf_.at(id).drone_id == planner_manager_->pp_.drone_id))
        {
          continue;
        }

        double t_X = t_cur_global - planner_manager_->swarm_trajs_buf_.at(id).start_time_.seconds();
        if (exec_state_ == INFLATED_ESCAPE) {
          t_X += t - t_cur;
          if (t_X < 0 || t_X > planner_manager_->swarm_trajs_buf_.at(id).duration_) continue;
        }
        Eigen::Vector3d swarm_pridicted = planner_manager_->swarm_trajs_buf_.at(id).position_traj_.evaluateDeBoorT(t_X);
        const Eigen::Vector3d ours = exec_state_ == INFLATED_ESCAPE
            ? Eigen::Vector3d(info->position_traj_.evaluateDeBoorT(t)) : p_cur;
        double dist = (ours - swarm_pridicted).norm();

        if (dist < CLEARANCE)
        {
          occ = true;
          break;
        }
      }

      if (occ)
      {
        if (exec_state_ == INFLATED_ESCAPE)
        {
          callEmergencyStop(odom_pos_);
          have_target_ = false;
          have_pending_goal_ = false;
          changeFSMExecState(OCCUPIED_START, "ESCAPE_COLLISION");
          return;
        }
        // A queued goal must not discard the escape from a known collision.
        if (planFromCurrentTraj(1, false)) // Make a chance
        {
          changeFSMExecState(EXEC_TRAJ, "SAFETY");
          publishSwarmTrajs(false);
          return;
        }
        else
        {
          if (t - t_cur < emergency_time_) // 0.8s of emergency time
          {
            RCLCPP_WARN(node_->get_logger(), "Suddenly discovered obstacles. emergency stop! time=%f", t - t_cur);

            // Send the stop now: the next exec tick drains queued goals
            // before its state switch and could otherwise erase this stop.
            if (exec_state_ != EMERGENCY_STOP || flag_escape_emergency_)
              callEmergencyStop(odom_pos_);
            flag_escape_emergency_ = false;
            changeFSMExecState(EMERGENCY_STOP, "SAFETY");
          }
          else
          {
            RCLCPP_WARN(node_->get_logger(), "current traj in collision, replan.");
            changeFSMExecState(REPLAN_TRAJ, "SAFETY");
          }
          return;
        }
        break;
      }
    }
  }

  bool EGOReplanFSM::callReboundReplan(bool flag_use_poly_init, bool flag_randomPolyTraj, bool preemptible)
  {
    if (preemptible && commandsPending()) return false;
    getLocalTarget();
    const auto previous_trajectory = planner_manager_->local_data_;

    bool plan_and_refine_success =
        planner_manager_->reboundReplan(start_pt_, start_vel_, start_acc_, local_target_pt_, local_target_vel_, (have_new_target_ || flag_use_poly_init), flag_randomPolyTraj);
    have_new_target_ = false;

    cout << "refine_success=" << plan_and_refine_success << endl;

    if (plan_and_refine_success)
    {
      // Fusion can proceed throughout optimization. Only this final check and
      // publication see a locked map; restore the flown trajectory on rejection.
      auto map_guard = planner_manager_->grid_map_->lock();
      if (planner_manager_->grid_map_->getOdomDepthTimeout() ||
          !trajectoryPublicationSafe(*planner_manager_->grid_map_, planner_manager_->local_data_.position_traj_))
      {
        planner_manager_->local_data_ = previous_trajectory;
        return false;
      }
      std::lock_guard<std::mutex> input_guard(input_mutex_);
      if (preemptible && !commands_.empty())
      {
        planner_manager_->local_data_ = previous_trajectory;
        return false;
      }
      auto info = &planner_manager_->local_data_;

      traj_utils::msg::Bspline bspline;
      bspline.order = 3;
      bspline.start_time = info->start_time_;
      bspline.traj_id = info->traj_id_;

      Eigen::MatrixXd pos_pts = info->position_traj_.getControlPoint();
      bspline.pos_pts.reserve(pos_pts.cols());
      for (int i = 0; i < pos_pts.cols(); ++i)
      {
        geometry_msgs::msg::Point pt;
        pt.x = pos_pts(0, i);
        pt.y = pos_pts(1, i);
        pt.z = pos_pts(2, i);
        bspline.pos_pts.push_back(pt);
      }

      Eigen::VectorXd knots = info->position_traj_.getKnot();

      bspline.knots.reserve(knots.rows());
      for (int i = 0; i < knots.rows(); ++i)
      {
        bspline.knots.push_back(knots(i));
      }

      /* 1. publish traj to traj_server */
      bspline_pub_->publish(bspline);

      /* 2. publish traj to the next drone of swarm */

      /* 3. publish traj for visualization */
      visualization_->displayOptimalList(info->position_traj_.get_control_points(), 0);
    }

    return plan_and_refine_success;
  }

  void EGOReplanFSM::publishSwarmTrajs(bool startup_pub)
  {
    auto info = &planner_manager_->local_data_;

    traj_utils::msg::Bspline bspline;
    bspline.order = 3;
    bspline.start_time = info->start_time_;
    bspline.drone_id = planner_manager_->pp_.drone_id;
    bspline.traj_id = info->traj_id_;

    Eigen::MatrixXd pos_pts = info->position_traj_.getControlPoint();
    bspline.pos_pts.reserve(pos_pts.cols());
    for (int i = 0; i < pos_pts.cols(); ++i)
    {
      geometry_msgs::msg::Point pt;
      pt.x = pos_pts(0, i);
      pt.y = pos_pts(1, i);
      pt.z = pos_pts(2, i);
      bspline.pos_pts.push_back(pt);
    }

    Eigen::VectorXd knots = info->position_traj_.getKnot();

    bspline.knots.reserve(knots.rows());
    for (int i = 0; i < knots.rows(); ++i)
    {
      bspline.knots.push_back(knots(i));
    }

    sendSwarmTrajs(bspline, startup_pub);
  }

  void EGOReplanFSM::sendSwarmTrajs(const traj_utils::msg::Bspline &bspline, bool startup_pub)
  {
    if (startup_pub)
    {
      // SwarmDeck: an absent predecessor leaves empty entries, which followers skip.
      multi_bspline_msgs_buf_.drone_id_from = planner_manager_->pp_.drone_id;
      multi_bspline_msgs_buf_.traj.resize(planner_manager_->pp_.drone_id);
      multi_bspline_msgs_buf_.traj.push_back(bspline);
      swarm_trajs_pub_->publish(multi_bspline_msgs_buf_);
    }

    broadcast_bspline_pub_->publish(bspline);
  }

  bool EGOReplanFSM::callEmergencyStop(Eigen::Vector3d stop_pos)
  {

    planner_manager_->EmergencyStop(stop_pos);

    publishLocalTrajectory();
    return true;
  }

  void EGOReplanFSM::publishLocalTrajectory()
  {
    auto info = &planner_manager_->local_data_;

    /* publish traj */
    traj_utils::msg::Bspline bspline;
    bspline.order = 3;
    bspline.start_time = info->start_time_;
    bspline.traj_id = info->traj_id_;

    Eigen::MatrixXd pos_pts = info->position_traj_.getControlPoint();
    bspline.pos_pts.reserve(pos_pts.cols());
    for (int i = 0; i < pos_pts.cols(); ++i)
    {
      geometry_msgs::msg::Point pt;
      pt.x = pos_pts(0, i);
      pt.y = pos_pts(1, i);
      pt.z = pos_pts(2, i);
      bspline.pos_pts.push_back(pt);
    }

    Eigen::VectorXd knots = info->position_traj_.getKnot();
    bspline.knots.reserve(knots.rows());
    for (int i = 0; i < knots.rows(); ++i)
    {
      bspline.knots.push_back(knots(i));
    }

    bspline_pub_->publish(bspline);

  }

  void EGOReplanFSM::getLocalTarget()
  {
    double t;

    double t_step = planning_horizen_ / 20 / planner_manager_->pp_.max_vel_;
    double dist_min = 9999, dist_min_t = 0.0;
    for (t = planner_manager_->global_data_.last_progress_time_; t < planner_manager_->global_data_.global_duration_; t += t_step)
    {
      Eigen::Vector3d pos_t = planner_manager_->global_data_.getPosition(t);
      double dist = (pos_t - start_pt_).norm();

      if (t < planner_manager_->global_data_.last_progress_time_ + 1e-5 && dist > planning_horizen_)
      {
        // Important cornor case!
        for (; t < planner_manager_->global_data_.global_duration_; t += t_step)
        {
          Eigen::Vector3d pos_t_temp = planner_manager_->global_data_.getPosition(t);
          double dist_temp = (pos_t_temp - start_pt_).norm();
          if (dist_temp < planning_horizen_)
          {
            pos_t = pos_t_temp;
            dist = (pos_t - start_pt_).norm();
            cout << "Escape cornor case \"getLocalTarget\"" << endl;
            break;
          }
        }
      }

      if (dist < dist_min)
      {
        dist_min = dist;
        dist_min_t = t;
      }

      if (dist >= planning_horizen_)
      {
        local_target_pt_ = pos_t;
        planner_manager_->global_data_.last_progress_time_ = dist_min_t;
        break;
      }
    }
    if (t > planner_manager_->global_data_.global_duration_) // Last global point
    {
      local_target_pt_ = end_pt_;
      planner_manager_->global_data_.last_progress_time_ = planner_manager_->global_data_.global_duration_;
    }

    if ((end_pt_ - local_target_pt_).norm() < (planner_manager_->pp_.max_vel_ * planner_manager_->pp_.max_vel_) / (2 * planner_manager_->pp_.max_acc_))
    {
      local_target_vel_ = Eigen::Vector3d::Zero();
    }
    else
    {
      local_target_vel_ = planner_manager_->global_data_.getVelocity(t);
    }
  }

} // namespace ego_planner
