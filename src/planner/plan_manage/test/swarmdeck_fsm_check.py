#!/usr/bin/env python3
"""SwarmDeck's checks of EGO's goal, cancel and FSM-state interface.

Each scenario starts real ego_planner_node and traj_server processes in the
SwarmDeck layout (namespace /<robot>/ego, relative topics remapped as
SwarmDeck's drone.launch.py does) and flies perfect-tracker drones: odometry
is the latest PositionCommand, and the lidar is a synthetic corridor cloud.
It then asserts on the FSM states, B-splines and commands the planners publish.

    source install/setup.bash
    python3 src/planner/plan_manage/test/swarmdeck_fsm_check.py [SCENARIO ...]

Run it in an isolated container (--network none, a private ROS_DOMAIN_ID).
Every scenario uses its own topic prefix, so scenarios cannot hear each other.
Exit status 0 when every scenario passes. EGO_CHECK_PLANNER_PREFIX runs the
planner under a wrapper, e.g. "valgrind --track-origins=yes" (ros2 run --prefix);
EGO_CHECK_READY_TIMEOUT (s, default 20) then gives the slower planner time to start.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point as PointMsg
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Empty, String
from traj_utils.msg import Bspline, MultiBsplines

LOG_DIR = os.environ.get("EGO_CHECK_LOG_DIR", tempfile.mkdtemp(prefix="ego-check-"))
READY_TIMEOUT = float(os.environ.get("EGO_CHECK_READY_TIMEOUT", "20"))


def planner_parameters(robot: str, drone_id: int) -> dict:
    """SwarmDeck's drone.launch.py values, at 2 m/s."""
    return {
        "fsm/flight_type": 1,
        "fsm/thresh_replan_time": 1.0,
        "fsm/thresh_no_replan_meter": 1.0,
        "fsm/planning_horizon": 7.5,
        "fsm/planning_horizen_time": 3.0,
        "fsm/emergency_time": 1.0,
        "fsm/realworld_experiment": False,
        "fsm/fail_safe": True,
        "fsm/waypoint_num": 1,
        "fsm/waypoint0_x": 0.0,
        "fsm/waypoint0_y": 0.0,
        "fsm/waypoint0_z": 1.0,
        "grid_map/resolution": 0.1,
        "grid_map/map_size_x": 50.0,
        "grid_map/map_size_y": 50.0,
        "grid_map/map_size_z": 6.0,
        "grid_map/local_update_range_x": 5.5,
        "grid_map/local_update_range_y": 5.5,
        "grid_map/local_update_range_z": 4.5,
        "grid_map/obstacles_inflation": 0.25,
        "grid_map/local_map_margin": 10,
        "grid_map/ground_height": -1.0,
        "grid_map/cx": 321.04638671875,
        "grid_map/cy": 243.44969177246094,
        "grid_map/fx": 387.229248046875,
        "grid_map/fy": 387.229248046875,
        "grid_map/use_depth_filter": True,
        "grid_map/depth_filter_tolerance": 0.15,
        "grid_map/depth_filter_maxdist": 5.0,
        "grid_map/depth_filter_mindist": 0.2,
        "grid_map/depth_filter_margin": 2,
        "grid_map/k_depth_scaling_factor": 1000.0,
        "grid_map/skip_pixel": 2,
        "grid_map/p_hit": 0.65,
        "grid_map/p_miss": 0.35,
        "grid_map/p_min": 0.12,
        "grid_map/p_max": 0.90,
        "grid_map/p_occ": 0.80,
        "grid_map/min_ray_length": 0.1,
        "grid_map/max_ray_length": 8.0,
        "grid_map/virtual_ceil_height": -1.0,
        "grid_map/visualization_truncate_height": 100.0,
        "grid_map/show_occ_time": False,
        "grid_map/pose_type": 2,
        "grid_map/frame_id": f"{robot}/odom",
        "grid_map/odom_depth_timeout": 3.0,
        "manager/max_vel": 2.0,
        "manager/max_acc": 3.0,
        "manager/max_jerk": 4.0,
        "manager/control_points_distance": 0.4,
        "manager/feasibility_tolerance": 0.05,
        "manager/planning_horizon": 7.5,
        "manager/use_distinctive_trajs": True,
        "manager/drone_id": drone_id,
        "optimization/lambda_smooth": 1.0,
        "optimization/lambda_collision": 0.5,
        "optimization/lambda_feasibility": 0.1,
        "optimization/lambda_fitness": 1.0,
        "optimization/dist0": 0.5,
        "optimization/swarm_clearance": 0.5,
        "optimization/max_vel": 2.0,
        "optimization/max_acc": 3.0,
        "bspline/limit_vel": 2.0,
        "bspline/limit_acc": 3.0,
        "bspline/limit_ratio": 1.1,
        "prediction/obj_num": 0,
        "prediction/lambda": 1.0,
        "prediction/predict_rate": 1.0,
    }


def ros_arg_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return repr(value) if isinstance(value, float) else str(value)


def corridor_cloud_points() -> np.ndarray:
    """Walls at y = +/-6 from x = -5 to 25, 0 to 3 m high, every 0.25 m."""
    xs = np.arange(-5.0, 25.0 + 1e-9, 0.25)
    zs = np.arange(0.0, 3.0 + 1e-9, 0.25)
    grid_x, grid_z = np.meshgrid(xs, zs)
    walls = [
        np.column_stack([grid_x.ravel(), np.full(grid_x.size, y), grid_z.ravel()])
        for y in (-6.0, 6.0)
    ]
    return np.vstack(walls).astype("<f4")


CLOUD_POINTS = corridor_cloud_points()
CLOUD_FIELDS = [
    PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
    for i, n in enumerate("xyz")
]


class Harness:
    """The rclpy side of every scenario: one node spun in a background thread."""

    def __init__(self) -> None:
        rclpy.init()
        self.node = rclpy.create_node("ego_fsm_check")
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()
        self.processes: list[subprocess.Popen] = []

    def now(self) -> float:
        return time.monotonic()

    def spawn(self, name: str, cmd: list[str]) -> subprocess.Popen:
        log = open(os.path.join(LOG_DIR, f"{name}.log"), "a")
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.processes.append(process)
        return process

    def stop_processes(self) -> None:
        for process in self.processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        deadline = time.monotonic() + max(5.0, READY_TIMEOUT / 2)
        for process in self.processes:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        self.processes.clear()

    def shutdown(self) -> None:
        self.stop_processes()
        self.executor.shutdown()
        self.thread.join(timeout=5.0)
        self.node.destroy_node()
        rclpy.shutdown()


def wait_for(predicate, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return bool(predicate())


class Drone:
    """One planner and traj_server pair and its perfect-tracker drone."""

    def __init__(self, harness: Harness, scenario: str, index: int, drone_id: int,
                 start=(0.0, 0.0, 1.2), extra_parameters: dict | None = None) -> None:
        self.harness = harness
        self.robot = f"{scenario}_{index}"
        self.shared = f"/{scenario}_ego"
        self.drone_id = drone_id
        self.pos = list(start)
        self.vel = [0.0, 0.0, 0.0]
        self.odom_enabled = False
        self.states: list[tuple[float, str]] = []
        self.bsplines: list[tuple[float, Bspline]] = []
        self.commands = 0
        self.track: list[list[float]] = []
        node = harness.node
        ns = f"/{self.robot}/ego"
        latched = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.subscriptions = [
            node.create_subscription(String, f"{ns}/planning/fsm_state", self.on_state, latched),
            node.create_subscription(Bspline, f"{ns}/planning/bspline", self.on_bspline, 10),
            node.create_subscription(
                PositionCommand, f"/{self.robot}/flight/position_cmd", self.on_command, 100
            ),
        ]
        self.goal_pub = node.create_publisher(PoseStamped, f"{ns}/goal", 10)
        self.cancel_pub = node.create_publisher(Empty, f"{ns}/cancel", 10)
        self.odom_pub = node.create_publisher(Odometry, f"/{self.robot}/odom", 10)
        self.cloud_pub = node.create_publisher(PointCloud2, f"/{self.robot}/scan/points_odom", 10)
        self.timers = [
            node.create_timer(0.01, self.publish_odom),
            node.create_timer(0.1, self.publish_cloud),
        ]
        self.extra_parameters = extra_parameters or {}
        self.processes: list[subprocess.Popen] = []
        self.start_processes(self.extra_parameters)

    # -- processes
    def restart(self) -> None:
        """Kill this drone's planner and traj_server and start them again."""
        for process in self.processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        for process in self.processes:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        self.processes = []
        self.start_processes(self.extra_parameters)

    def start_processes(self, extra_parameters: dict) -> None:
        ns = f"/{self.robot}/ego"
        parameters = planner_parameters(self.robot, self.drone_id)
        parameters.update(extra_parameters)
        swarm = "drone_{}_planning/swarm_trajs"
        remaps = {
            "odom_world": f"/{self.robot}/odom",
            "grid_map/odom": f"/{self.robot}/odom",
            "grid_map/cloud": f"/{self.robot}/scan/points_odom",
            "planning/broadcast_bspline_from_planner": f"{self.shared}/broadcast_bspline",
            "planning/broadcast_bspline_to_planner": f"{self.shared}/broadcast_bspline",
            swarm.format(self.drone_id): f"{self.shared}/{swarm.format(self.drone_id)}",
        }
        if self.drone_id >= 1:
            previous = swarm.format(self.drone_id - 1)
            remaps[previous] = f"{self.shared}/{previous}"
        prefix = os.environ.get("EGO_CHECK_PLANNER_PREFIX")
        cmd = ["ros2", "run"] + (["--prefix", prefix] if prefix else [])
        cmd += ["ego_planner", "ego_planner_node", "--ros-args",
               "-r", "__node:=ego_planner", "-r", f"__ns:={ns}"]
        for key, value in remaps.items():
            cmd += ["-r", f"{key}:={value}"]
        for key, value in parameters.items():
            cmd += ["-p", f"{key}:={ros_arg_value(value)}"]
        self.processes.append(self.harness.spawn(f"{self.robot}_planner", cmd))
        self.processes.append(self.harness.spawn(f"{self.robot}_traj_server", [
            "ros2", "run", "ego_planner", "traj_server", "--ros-args",
            "-r", "__node:=traj_server", "-r", f"__ns:={ns}",
            "-r", f"position_cmd:=/{self.robot}/flight/position_cmd",
            "-p", "traj_server/time_forward:=1.0",
        ]))

    def close(self) -> None:
        node = self.harness.node
        for timer in self.timers:
            node.destroy_timer(timer)
        for subscription in self.subscriptions:
            node.destroy_subscription(subscription)
        for publisher in (self.goal_pub, self.cancel_pub, self.odom_pub, self.cloud_pub):
            node.destroy_publisher(publisher)

    # -- callbacks
    def on_state(self, msg: String) -> None:
        self.states.append((self.harness.now(), msg.data))

    def on_bspline(self, msg: Bspline) -> None:
        self.bsplines.append((self.harness.now(), msg))

    def on_command(self, msg: PositionCommand) -> None:
        self.commands += 1
        self.track.append([msg.position.x, msg.position.y, msg.position.z])
        self.pos = [msg.position.x, msg.position.y, msg.position.z]
        self.vel = [msg.velocity.x, msg.velocity.y, msg.velocity.z]

    def publish_odom(self) -> None:
        if not self.odom_enabled:
            return
        odom = Odometry()
        odom.header.stamp = self.harness.node.get_clock().now().to_msg()
        odom.header.frame_id = f"{self.robot}/odom"
        odom.child_frame_id = f"{self.robot}/base_link"
        p, v = odom.pose.pose.position, odom.twist.twist.linear
        p.x, p.y, p.z = self.pos
        v.x, v.y, v.z = self.vel
        odom.pose.pose.orientation.w = 1.0
        self.odom_pub.publish(odom)

    def publish_cloud(self) -> None:
        cloud = PointCloud2()
        cloud.header.stamp = self.harness.node.get_clock().now().to_msg()
        cloud.header.frame_id = f"{self.robot}/odom"
        cloud.height, cloud.width = 1, len(CLOUD_POINTS)
        cloud.fields = CLOUD_FIELDS
        cloud.is_bigendian = False
        cloud.point_step, cloud.row_step = 12, 12 * len(CLOUD_POINTS)
        cloud.is_dense = True
        cloud.data = CLOUD_POINTS.tobytes()
        self.cloud_pub.publish(cloud)

    # -- commands and queries
    def wait_ready(self, timeout: float = READY_TIMEOUT) -> bool:
        """The planner is up and subscribed to its goal and cancel topics."""
        return wait_for(
            lambda: self.states
            and self.goal_pub.get_subscription_count() > 0
            and self.cancel_pub.get_subscription_count() > 0,
            timeout,
        )

    def send_goal(self, x: float, y: float, z: float) -> None:
        goal = PoseStamped()
        goal.header.stamp = self.harness.node.get_clock().now().to_msg()
        goal.header.frame_id = f"{self.robot}/odom"
        goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = x, y, z
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)

    def cancel(self) -> None:
        self.cancel_pub.publish(Empty())

    def state(self) -> str | None:
        return self.states[-1][1] if self.states else None

    def reached_state(self, name: str, since: float = 0.0) -> float | None:
        """When the planner first reported `name` after `since`, or None."""
        for stamp, state in self.states:
            if stamp >= since and state == name:
                return stamp
        return None

    def near(self, point, tolerance: float) -> bool:
        return math.dist(self.pos, point) <= tolerance

    def log(self) -> str:
        with open(os.path.join(LOG_DIR, f"{self.robot}_planner.log")) as stream:
            return stream.read()

    def bsplines_since(self, since: float) -> list[Bspline]:
        return [msg for stamp, msg in self.bsplines if stamp >= since]


def is_stationary(msg: Bspline) -> bool:
    points = np.array([[p.x, p.y, p.z] for p in msg.pos_pts])
    return len(points) > 0 and float(np.abs(points - points[0]).max()) < 1e-6


def spline_point(msg: Bspline) -> list[float]:
    return [msg.pos_pts[0].x, msg.pos_pts[0].y, msg.pos_pts[0].z]


class Peer:
    """Another drone's broadcast trajectory: hovering at `point` from now."""

    def __init__(self, harness: Harness, shared: str, drone_id: int) -> None:
        self.harness = harness
        self.drone_id = drone_id
        self.pub = harness.node.create_publisher(Bspline, f"{shared}/broadcast_bspline", 10)

    def hover(self, point, points: int = 6, age: float = 0.0) -> Bspline:
        """A hover at `point` that started `age` seconds ago and lasts about
        `points` - 2 seconds."""
        msg = Bspline()
        msg.order = 3
        msg.drone_id = self.drone_id
        msg.traj_id = 1
        start = self.harness.node.get_clock().now() - Duration(seconds=age)
        msg.start_time = start.to_msg()
        interval = 1.0
        for _ in range(points):
            msg.pos_pts.append(PointMsg(x=float(point[0]), y=float(point[1]), z=float(point[2])))
        msg.knots = [(i - msg.order) * interval for i in range(len(msg.pos_pts) + msg.order + 1)]
        return msg

    def hover_at(self, point, points: int = 6) -> None:
        self.pub.publish(self.hover(point, points))

    def close(self) -> None:
        self.harness.node.destroy_publisher(self.pub)


class StartupChain:
    """A drone's retained startup handshake on drone_<id>_planning/swarm_trajs."""

    def __init__(self, harness: Harness, shared: str, drone_id: int) -> None:
        self.harness = harness
        self.drone_id = drone_id
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub = harness.node.create_publisher(
            MultiBsplines, f"{shared}/drone_{drone_id}_planning/swarm_trajs", latched
        )

    def announce(self, trajectories: list[Bspline]) -> None:
        msg = MultiBsplines()
        msg.drone_id_from = self.drone_id
        msg.traj = trajectories
        self.pub.publish(msg)

    def close(self) -> None:
        self.harness.node.destroy_publisher(self.pub)


def min_distance(track: list[list[float]], point) -> float:
    return min((math.dist(p, point) for p in track), default=float("inf"))


class Scenario:
    def __init__(self, harness: Harness, name: str) -> None:
        self.harness = harness
        self.name = name
        self.drones: list[Drone] = []
        self.failures: list[str] = []
        self.notes: list[str] = []

    def drone(self, drone_id: int, **kwargs) -> Drone:
        drone = Drone(self.harness, self.name, len(self.drones), drone_id, **kwargs)
        self.drones.append(drone)
        return drone

    def check(self, condition: bool, failure: str) -> bool:
        if not condition:
            self.failures.append(failure)
        return condition

    def note(self, text: str) -> None:
        self.notes.append(text)

    def close(self) -> None:
        self.harness.stop_processes()
        for drone in self.drones:
            drone.close()


# ------------------------------------------------------------------ scenarios


def early_goal(s: Scenario) -> None:
    """A goal before odometry waits for it; a newer goal replaces it."""
    d = s.drone(0)
    if not s.check(d.wait_ready(), "planner not up"):
        return
    d.send_goal(12.0, 0.0, 1.5)
    time.sleep(1.5)
    s.check(not d.bsplines, "planned a trajectory before odometry")
    s.check(d.state() == "INIT", f"state {d.state()} before odometry, expected INIT")
    replacement = (6.0, 2.0, 1.8)
    d.send_goal(*replacement)
    time.sleep(0.5)
    s.check(not d.bsplines, "planned the replacement goal before odometry")
    started = d.harness.now()
    d.odom_enabled = True
    executing = wait_for(lambda: d.reached_state("EXEC_TRAJ", started), 10.0)
    s.check(executing, "the pending goal was not flown after odometry arrived")
    arrived = wait_for(lambda: d.near(replacement, 0.3), 20.0)
    s.check(arrived, f"did not reach the replacement goal {replacement}: at {d.pos}")
    s.note(f"reached {replacement}: {arrived}, at {[round(v, 2) for v in d.pos]}")


def early_goal_cancel(s: Scenario) -> None:
    """A goal cancelled before odometry is never flown."""
    d = s.drone(0)
    if not s.check(d.wait_ready(), "planner not up"):
        return
    d.send_goal(12.0, 0.0, 1.5)
    time.sleep(1.0)
    d.cancel()
    time.sleep(0.5)
    d.odom_enabled = True
    s.check(wait_for(lambda: d.state() == "WAIT_TARGET", 5.0), "never reached WAIT_TARGET")
    time.sleep(4.0)
    s.check(d.state() == "WAIT_TARGET", f"state {d.state()} after a cancelled goal")
    s.check(not d.bsplines, f"{len(d.bsplines)} trajectories for a cancelled goal")
    s.check(d.commands == 0, f"{d.commands} commands for a cancelled goal")


def cancel_stops(s: Scenario) -> None:
    """Cancel stops the drone where it is, and a conflicting peer trajectory
    afterwards does not make the planner fly toward the cancelled goal."""
    d = s.drone(0)
    peer = Peer(s.harness, d.shared, 7)
    try:
        if not s.check(d.wait_ready(), "planner not up"):
            return
        d.odom_enabled = True
        s.check(wait_for(lambda: d.state() == "WAIT_TARGET", 10.0), "never reached WAIT_TARGET")
        goal = (18.0, 0.0, 1.5)
        d.send_goal(*goal)
        if not s.check(wait_for(lambda: d.reached_state("EXEC_TRAJ"), 10.0), "never flew"):
            return
        s.check(wait_for(lambda: d.pos[0] > 3.0, 10.0), f"did not get going: at {d.pos}")

        # Control: in flight, the same kind of peer trajectory is a conflict.
        mark = len(d.log())
        peer.hover_at([d.pos[0] + 1.5, d.pos[1], d.pos[2]])
        time.sleep(1.0)
        s.check("[TRAJ_CHECK]" in d.log()[mark:], "control: an in-flight conflict was not replanned")

        s.check(wait_for(lambda: d.pos[0] > 6.0, 10.0), f"did not get past x 6: at {d.pos}")
        cancelled = s.harness.now()
        at_cancel = list(d.pos)
        mark = len(d.log())
        d.cancel()
        s.check(wait_for(lambda: d.bsplines_since(cancelled), 2.0), "no stop trajectory after cancel")
        after = d.bsplines_since(cancelled)
        stop = after[0] if after else None
        s.check(stop is not None and is_stationary(stop), "the first trajectory after cancel is not a stop")
        if stop is not None:
            s.check(math.dist(spline_point(stop), at_cancel) < 0.5,
                    f"stop at {spline_point(stop)}, drone was at {at_cancel}")
        s.check(wait_for(lambda: d.state() == "WAIT_TARGET", 2.0), f"state {d.state()} after cancel")
        time.sleep(0.5)
        stopped_at = list(d.pos)
        count = len(after)
        for _ in range(30):  # 3 s of fresh conflicting peer trajectories on the drone
            peer.hover_at(d.pos)
            time.sleep(0.1)
        time.sleep(1.0)
        s.check("[TRAJ_CHECK]" not in d.log()[mark:], "a peer trajectory replanned after cancel")
        s.check(len(d.bsplines_since(cancelled)) == count,
                f"{len(d.bsplines_since(cancelled)) - count} new trajectories after cancel")
        s.check(d.state() == "WAIT_TARGET", f"state {d.state()} after the peer trajectories")
        s.check(math.dist(d.pos, stopped_at) < 0.1, f"moved from {stopped_at} to {d.pos} after cancel")
        s.note(f"cancelled at x {at_cancel[0]:.2f}, stopped at x {stopped_at[0]:.2f}, "
               f"now x {d.pos[0]:.2f}; goal x {goal[0]}")
    finally:
        peer.close()


STARTUP_TIMEOUT_WARNING = "starting without it"


def two_drones_staggered(s: Scenario) -> None:
    """Drone 1 gets its goal before its odometry, drone 0 gets its goal in
    WAIT_TARGET after drone 1's odometry: both fly, with no startup timeout."""
    d0 = s.drone(0, start=(0.0, 0.0, 1.2))
    d1 = s.drone(1, start=(0.0, 3.0, 1.2))
    if not s.check(d0.wait_ready() and d1.wait_ready(), "planners not up"):
        return
    goal1, goal0 = (10.0, 3.0, 1.5), (10.0, 0.0, 1.5)
    d1.send_goal(*goal1)
    time.sleep(1.0)
    d0.odom_enabled = True
    s.check(wait_for(lambda: d0.state() == "WAIT_TARGET", 5.0), f"drone 0: {d0.state()}")
    time.sleep(1.0)
    odom1 = s.harness.now()
    d1.odom_enabled = True
    time.sleep(1.0)
    goal0_sent = s.harness.now()
    s.check(d0.state() == "WAIT_TARGET", f"drone 0 in {d0.state()} before its goal")
    d0.send_goal(*goal0)
    exec1 = wait_for(lambda: d1.reached_state("EXEC_TRAJ", odom1), 8.0)
    exec0 = wait_for(lambda: d0.reached_state("EXEC_TRAJ", goal0_sent), 8.0)
    s.check(exec1, "drone 1 never reached EXEC_TRAJ")
    s.check(exec0, "drone 0 never reached EXEC_TRAJ")
    if exec1:
        s.note(f"drone 1 EXEC_TRAJ {d1.reached_state('EXEC_TRAJ', odom1) - odom1:.2f} s after its odometry")
    if exec0:
        s.note(f"drone 0 EXEC_TRAJ {d0.reached_state('EXEC_TRAJ', goal0_sent) - goal0_sent:.2f} s after its goal")
    s.check(wait_for(lambda: d0.near(goal0, 0.3) and d1.near(goal1, 0.3), 20.0),
            f"goals not reached: drone 0 at {d0.pos}, drone 1 at {d1.pos}")
    s.check(STARTUP_TIMEOUT_WARNING not in d1.log(), "drone 1 timed out waiting for drone 0")


def follower_waits_for_leader(s: Scenario) -> None:
    """Drone 1 has odometry and a goal before drone 0 is up: it waits in
    SEQUENTIAL_START and flies as soon as drone 0 has odometry."""
    d0 = s.drone(0, start=(0.0, 0.0, 1.2))
    d1 = s.drone(1, start=(0.0, 3.0, 1.2))
    if not s.check(d0.wait_ready() and d1.wait_ready(), "planners not up"):
        return
    goal1 = (10.0, 3.0, 1.5)
    d1.odom_enabled = True
    s.check(wait_for(lambda: d1.state() == "WAIT_TARGET", 5.0), f"drone 1: {d1.state()}")
    d1.send_goal(*goal1)
    s.check(wait_for(lambda: d1.state() == "SEQUENTIAL_START", 2.0), f"drone 1 in {d1.state()}")
    time.sleep(3.0)
    s.check(d1.state() == "SEQUENTIAL_START", f"drone 1 left SEQUENTIAL_START for {d1.state()} alone")
    s.check(not d1.bsplines, "drone 1 planned before drone 0 was up")
    odom0 = s.harness.now()
    d0.odom_enabled = True
    s.check(wait_for(lambda: d1.reached_state("EXEC_TRAJ", odom0), 3.0),
            "drone 1 did not fly within 3 s of drone 0's odometry")
    if d1.reached_state("EXEC_TRAJ", odom0):
        s.note(f"drone 1 EXEC_TRAJ {d1.reached_state('EXEC_TRAJ', odom0) - odom0:.2f} s after drone 0's odometry")
    s.check(wait_for(lambda: d1.near(goal1, 0.3), 20.0), f"drone 1 at {d1.pos}, goal {goal1}")
    s.check(STARTUP_TIMEOUT_WARNING not in d1.log(), "drone 1 timed out waiting for drone 0")
    d0.send_goal(10.0, 0.0, 1.5)
    s.check(wait_for(lambda: d0.reached_state("EXEC_TRAJ"), 5.0), "drone 0 never flew")


def lone_follower(s: Scenario) -> None:
    """Drone 1 with no drone 0 running starts after fsm/sequential_start_timeout_s."""
    d1 = s.drone(1, start=(0.0, 3.0, 1.2))
    if not s.check(d1.wait_ready(), "planner not up"):
        return
    odom1 = s.harness.now()
    d1.odom_enabled = True
    time.sleep(1.0)
    goal1 = (10.0, 3.0, 1.5)
    d1.send_goal(*goal1)
    s.check(wait_for(lambda: d1.state() == "SEQUENTIAL_START", 2.0), f"drone 1 in {d1.state()}")
    s.check(wait_for(lambda: d1.reached_state("EXEC_TRAJ"), 15.0), "drone 1 never flew")
    started = d1.reached_state("EXEC_TRAJ")
    if started:
        waited = started - odom1
        s.note(f"drone 1 EXEC_TRAJ {waited:.2f} s after its odometry (timeout 10 s)")
        s.check(9.5 <= waited <= 12.0, f"drone 1 flew {waited:.2f} s after odometry, expected ~10 s")
    s.check(STARTUP_TIMEOUT_WARNING in d1.log(), "no startup-timeout warning")
    s.check(wait_for(lambda: d1.near(goal1, 0.3), 20.0), f"drone 1 at {d1.pos}, goal {goal1}")


def state_gaps(d: Drone, start: float, end: float) -> list[float]:
    stamps = [stamp for stamp, _ in d.states if start <= stamp <= end]
    return [b - a for a, b in zip([start] + stamps, stamps + [end])]


def heartbeat(s: Scenario) -> None:
    """fsm_state comes at least once a second, idle and while planning fails."""
    d = s.drone(0)
    if not s.check(d.wait_ready(), "planner not up"):
        return
    d.odom_enabled = True
    s.check(wait_for(lambda: d.state() == "WAIT_TARGET", 5.0), f"state {d.state()}")
    start = s.harness.now()
    time.sleep(5.0)
    gaps = state_gaps(d, start, s.harness.now())
    s.note(f"idle: {len(gaps) - 1} messages in 5 s, max gap {max(gaps):.2f} s")
    s.check(max(gaps) <= 1.25, f"idle: fsm_state gap {max(gaps):.2f} s")

    mark = len(d.log())
    d.send_goal(3.0, 0.0, -4.0)  # below the map: planning fails once the drone nears the floor
    start = s.harness.now()
    time.sleep(8.0)
    end = s.harness.now()
    gaps = state_gaps(d, start, end)
    failures = d.log()[mark:].count("refine_success=0")
    s.note(f"failing plans: {failures} failures, {len(gaps) - 1} messages in 8 s, max gap {max(gaps):.2f} s")
    s.check(failures > 0, "control: planning did not fail")
    s.check(max(gaps) <= 1.25, f"while planning fails: fsm_state gap {max(gaps):.2f} s")


def handshake_keeps_live_peers(s: Scenario) -> None:
    """A predecessor's retained startup handshake, older than the live
    trajectories already received, neither replaces them nor drops other peers:
    the drone flies around both live hovers on its path."""
    d = s.drone(1, start=(0.0, 3.0, 1.2))
    peer0, peer5 = Peer(s.harness, d.shared, 0), Peer(s.harness, d.shared, 5)
    chain0 = StartupChain(s.harness, d.shared, 0)
    try:
        if not s.check(d.wait_ready(), "planner not up"):
            return
        d.odom_enabled = True
        s.check(wait_for(lambda: d.state() == "WAIT_TARGET", 5.0), f"state {d.state()}")
        live0, live5 = (4.0, 3.2, 1.2), (8.0, 2.8, 1.2)
        peer0.hover_at(live0, points=40)  # fresh, about 38 s long
        peer5.hover_at(live5, points=40)
        time.sleep(0.5)
        mark = len(d.log())
        chain0.announce([peer0.hover((1.0, 3.0, 1.2), age=30.0)])  # drone 0's startup, 30 s old
        s.check(wait_for(lambda: "kept the live one" in d.log()[mark:], 3.0),
                "the older handshake entry was not refused")
        s.check(STARTUP_TIMEOUT_WARNING not in d.log(), "the handshake was not taken")
        goal = (12.0, 3.0, 1.2)
        d.send_goal(*goal)
        s.check(wait_for(lambda: d.reached_state("EXEC_TRAJ"), 5.0), "never flew")
        s.check(wait_for(lambda: d.pos[0] > 9.5, 30.0), f"did not get past both hovers: at {d.pos}")
        near0, near5 = min_distance(d.track, live0), min_distance(d.track, live5)
        s.note(f"closest approach: drone 0's live hover {near0:.2f} m, drone 5's {near5:.2f} m")
        s.check(near0 >= 0.5, f"flew {near0:.2f} m from drone 0's live hover")
        s.check(near5 >= 0.5, f"flew {near5:.2f} m from drone 5's live hover")
    finally:
        peer0.close()
        peer5.close()
        chain0.close()


def blocked_start(s: Scenario) -> None:
    """A drone whose start is outside the map cannot plan its first trajectory:
    it reports the failure and never flies, and its startup handshake is not
    taken for a flown trajectory (no safety check, no emergency stop)."""
    d = s.drone(0, start=(0.0, 0.0, 5.5))  # the map spans z -1 to 5
    if not s.check(d.wait_ready(), "planner not up"):
        return
    d.odom_enabled = True
    s.check(wait_for(lambda: d.state() == "WAIT_TARGET", 5.0), f"state {d.state()}")
    time.sleep(1.0)
    d.send_goal(10.0, 0.0, 1.5)
    time.sleep(6.0)
    log = d.log()
    visited = sorted({state for _, state in d.states})
    s.note(f"states {visited}, {log.count('refine_success=0')} failed plans, "
           f"{len(d.bsplines)} trajectories, {d.commands} commands")
    s.check(log.count("refine_success=0") > 0, "control: the first plan did not fail")
    s.check("EMERGENCY_STOP" not in visited, "went to EMERGENCY_STOP before any flown trajectory")
    s.check("[SAFETY]" not in log, "the safety check ran before any flown trajectory")
    s.check(not d.bsplines, f"{len(d.bsplines)} trajectories sent to traj_server")
    s.check(d.commands == 0, f"{d.commands} commands: the drone was flown")
    s.check(d.state() == "GEN_NEW_TRAJ", f"state {d.state()}, expected GEN_NEW_TRAJ (failing)")


def goal_changes_in_sequential_start(s: Scenario) -> None:
    """While drone 1 waits in SEQUENTIAL_START (no drone 0): a replacement goal
    replaces the target without restarting the timeout, and after a cancel the
    handshake still completes and a later goal flies at once."""
    d = s.drone(1, start=(0.0, 3.0, 1.2))
    if not s.check(d.wait_ready(), "planner not up"):
        return
    odom = s.harness.now()
    d.odom_enabled = True
    s.check(wait_for(lambda: d.state() == "WAIT_TARGET", 5.0), f"state {d.state()}")
    d.send_goal(10.0, 3.0, 1.5)
    s.check(wait_for(lambda: d.state() == "SEQUENTIAL_START", 2.0), f"state {d.state()}")
    time.sleep(3.0)
    replacement = (6.0, 1.0, 1.8)
    d.send_goal(*replacement)
    time.sleep(1.0)
    s.check(d.state() == "SEQUENTIAL_START", f"state {d.state()} after the replacement goal")
    s.check(not d.bsplines, "planned before the handshake")
    s.check(wait_for(lambda: d.reached_state("EXEC_TRAJ"), 12.0), "never flew")
    started = d.reached_state("EXEC_TRAJ")
    if started:
        s.note(f"replacement: EXEC_TRAJ {started - odom:.2f} s after odometry (timeout 10 s, not restarted)")
        s.check(started - odom <= 11.0, f"flew {started - odom:.2f} s after odometry: timeout restarted")
    s.check(wait_for(lambda: d.near(replacement, 0.3), 20.0), f"at {d.pos}, goal {replacement}")

    # A second drone 1, cancelled while it waits.
    d2 = s.drone(1, start=(0.0, -3.0, 1.2))
    if not s.check(d2.wait_ready(), "second planner not up"):
        return
    odom2 = s.harness.now()
    d2.odom_enabled = True
    s.check(wait_for(lambda: d2.state() == "WAIT_TARGET", 5.0), f"state {d2.state()}")
    d2.send_goal(10.0, -3.0, 1.5)
    s.check(wait_for(lambda: d2.state() == "SEQUENTIAL_START", 2.0), f"state {d2.state()}")
    time.sleep(1.0)
    d2.cancel()
    s.check(wait_for(lambda: d2.state() == "WAIT_TARGET", 2.0), f"state {d2.state()} after cancel")
    s.check(wait_for(lambda: STARTUP_TIMEOUT_WARNING in d2.log(), 12.0), "the handshake did not complete")
    s.check(not d2.bsplines and d2.commands == 0, "flew the cancelled goal")
    s.check(d2.state() == "WAIT_TARGET", f"state {d2.state()} after the handshake")
    goal_sent = s.harness.now()
    d2.send_goal(6.0, -3.0, 1.5)
    s.check(wait_for(lambda: d2.reached_state("EXEC_TRAJ", goal_sent), 2.0), "a goal after the handshake did not fly at once")
    s.check(not d2.reached_state("SEQUENTIAL_START", goal_sent), "went back to SEQUENTIAL_START")
    s.note(f"cancelled: handshake by timeout {s.harness.now() - odom2:.1f} s after odometry, then flew on the next goal")


def predecessor_restart(s: Scenario) -> None:
    """Drone 0 restarts after the handshake: drone 1 keeps flying, does not
    wait for it again, and both fly new goals."""
    d0 = s.drone(0, start=(0.0, 0.0, 1.2))
    d1 = s.drone(1, start=(0.0, 3.0, 1.2))
    if not s.check(d0.wait_ready() and d1.wait_ready(), "planners not up"):
        return
    d0.odom_enabled = d1.odom_enabled = True
    s.check(wait_for(lambda: d1.state() == "WAIT_TARGET", 5.0), f"drone 1: {d1.state()}")
    d1.send_goal(18.0, 3.0, 1.5)
    s.check(wait_for(lambda: d1.reached_state("EXEC_TRAJ"), 5.0), "drone 1 never flew")
    s.check(wait_for(lambda: d1.pos[0] > 2.0, 5.0), f"drone 1 at {d1.pos}")
    restarted = s.harness.now()
    d0.restart()
    s.check(wait_for(lambda: d0.reached_state("WAIT_TARGET", restarted), 15.0), "drone 0 did not come back")
    time.sleep(1.0)
    s.check(not d1.reached_state("SEQUENTIAL_START", restarted), "drone 1 went back to SEQUENTIAL_START")
    s.check(d1.state() in ("EXEC_TRAJ", "REPLAN_TRAJ", "WAIT_TARGET"), f"drone 1 in {d1.state()}")
    s.check(wait_for(lambda: d1.near((18.0, 3.0, 1.5), 0.3), 20.0), f"drone 1 at {d1.pos}")
    goal_sent = s.harness.now()
    d1.send_goal(10.0, 3.0, 1.5)
    d0.send_goal(10.0, 0.0, 1.5)
    s.check(wait_for(lambda: d1.reached_state("EXEC_TRAJ", goal_sent), 3.0), "drone 1 did not fly a new goal")
    s.check(wait_for(lambda: d0.reached_state("EXEC_TRAJ", goal_sent), 3.0), "restarted drone 0 did not fly")
    s.check(STARTUP_TIMEOUT_WARNING not in d1.log(), "drone 1 timed out")


SCENARIOS = {
    "early_goal": early_goal,
    "early_goal_cancel": early_goal_cancel,
    "cancel_stops": cancel_stops,
    "two_drones_staggered": two_drones_staggered,
    "follower_waits_for_leader": follower_waits_for_leader,
    "lone_follower": lone_follower,
    "handshake_keeps_live_peers": handshake_keeps_live_peers,
    "blocked_start": blocked_start,
    "goal_changes_in_sequential_start": goal_changes_in_sequential_start,
    "predecessor_restart": predecessor_restart,
    "heartbeat": heartbeat,
}


def main(argv: list[str]) -> int:
    names = argv or list(SCENARIOS)
    harness = Harness()
    failed = []
    try:
        for name in names:
            scenario = Scenario(harness, name)
            began = time.monotonic()
            try:
                SCENARIOS[name](scenario)
            finally:
                scenario.close()
            verdict = "FAIL" if scenario.failures else "PASS"
            print(f"{verdict} {name} ({time.monotonic() - began:.1f} s)", flush=True)
            for line in scenario.notes:
                print(f"    note: {line}", flush=True)
            for line in scenario.failures:
                print(f"    {line}", flush=True)
            if scenario.failures:
                failed.append(name)
    finally:
        harness.shutdown()
    print(f"logs: {LOG_DIR}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
