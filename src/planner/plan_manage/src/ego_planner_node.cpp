#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <iostream>
#include <thread>

#include <ego_planner/ego_replan_fsm.h>

using namespace ego_planner;

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("ego_planner_node");

  EGOReplanFSM rebo_replan;

  rebo_replan.init(node);

  // Dedicated executors, not just a shared worker pool: a ready planning
  // timer can never consume the thread reserved for commands or lidar fusion.
  rclcpp::executors::SingleThreadedExecutor planning, sensing, input;
  planning.add_callback_group(node->get_node_base_interface()->get_default_callback_group(),
                              node->get_node_base_interface());
  sensing.add_callback_group(rebo_replan.sensingGroup(), node->get_node_base_interface());
  input.add_callback_group(rebo_replan.inputGroup(), node->get_node_base_interface());
  std::thread sensing_thread([&]() { sensing.spin(); });
  std::thread input_thread([&]() { input.spin(); });
  planning.spin();
  sensing.cancel();
  input.cancel();
  sensing_thread.join();
  input_thread.join();
  rclcpp::shutdown();

  return 0;
}

// #include <ros/ros.h>
// #include <csignal>
// #include <visualization_msgs/Marker.h>

// #include <plan_manage/ego_replan_fsm.h>

// using namespace ego_planner;

// void SignalHandler(int signal) {
//   if(ros::isInitialized() && ros::isStarted() && ros::ok() && !ros::isShuttingDown()){
//     ros::shutdown();
//   }
// }

// int main(int argc, char **argv) {

//   signal(SIGINT, SignalHandler);
//   signal(SIGTERM,SignalHandler);

//   ros::init(argc, argv, "ego_planner_node", ros::init_options::NoSigintHandler);
//   ros::NodeHandle nh("~");

//   EGOReplanFSM rebo_replan;

//   rebo_replan.init(nh);

//   // ros::Duration(1.0).sleep();
//   ros::AsyncSpinner async_spinner(4);
//   async_spinner.start();
//   ros::waitForShutdown();

//   return 0;
// }