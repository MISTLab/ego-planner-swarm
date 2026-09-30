#!/usr/bin/env python3
"""
Real-node command ordering and map progress regression; no simulator required.

Run beside swarmdeck_fsm_check.py in the built workspace, isolated ROS domain.
The unreachable goal drives real A* failures, not a test-only planner delay.
"""
import time

import numpy as np
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Empty
from swarmdeck_fsm_check import Harness, is_stationary, Scenario, wait_for


def run(s):
    """Check ordering, sensing progress, and avoidance of a late obstacle."""
    try:
        from traj_utils.msg import CommandState, PlannerCommand
    except ImportError:
        s.check(False, 'ordered command/attributed state interface is absent')
        return
    d = s.drone(0, extra_parameters={'fsm/ordered_commands': True})
    d.track_commands = False
    ns = f'/{d.robot}/ego'
    states, maps = [], []
    node = s.harness.node
    pub = node.create_publisher(
        PlannerCommand, ns + '/command', QoSProfile(history=HistoryPolicy.KEEP_ALL))
    state_sub = node.create_subscription(
        CommandState, ns + '/planning/command_state',
        lambda m: states.append((time.monotonic(), m)),
        QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    map_sub = node.create_subscription(
        PointCloud2, ns + '/grid_map/occupancy',
        lambda m: maps.append((time.monotonic(), m)), 10)

    def send(seq, goal=None):
        msg = PlannerCommand(
            sequence=seq, command=PlannerCommand.CANCEL if goal is None else PlannerCommand.GOAL)
        if goal is not None:
            msg.goal.pose.position.x, msg.goal.pose.position.y, msg.goal.pose.position.z = goal
        pub.publish(msg)

    try:
        if not s.check(
                wait_for(lambda: pub.get_subscription_count() > 0 and states, 20),
                'planner not ready'):
            return
        d.odom_enabled = True
        # A closed box: the goal lies outside; no local trajectory can reach it.
        axis = np.arange(-2, 2.01, .15)
        a, b = np.meshgrid(axis, axis)
        walls = []
        for x in (-2., 2.):
            walls.append(np.column_stack((np.full(a.size, x), a.ravel(), b.ravel() + 1.2)))
        for y in (-2., 2.):
            walls.append(np.column_stack((a.ravel(), np.full(a.size, y), b.ravel() + 1.2)))
        for z in (-.8, 3.2):
            walls.append(np.column_stack((a.ravel(), b.ravel(), np.full(a.size, z))))
        d.cloud_points = np.vstack(walls).astype('<f4')
        time.sleep(1.5)
        log_start = len(d.log())
        send(1, (5., 0., 1.2))
        s.check(wait_for(lambda: 'a star error' in d.log()[log_start:], 10),
                'fixture did not exercise failing A*')
        # A new occupied probe must appear while this unreachable goal is busy.
        began = time.monotonic()
        probe = np.array([[.55, .55, 1.25]], dtype='<f4')
        # Publish just the probe so background wall rays cannot vote misses
        # through its voxel. The already-fused closed box remains occupied.
        d.cloud_points = probe

        def fused():
            for stamp, cloud in maps:
                if stamp < began:
                    continue
                raw = np.frombuffer(bytes(cloud.data), dtype=np.uint8)
                raw = raw.reshape(-1, cloud.point_step)
                xyz = raw[:, :12].copy().view('<f4').reshape(-1, 3)
                if np.any(np.linalg.norm(xyz - probe[0], axis=1) < .12):
                    return True
            return False

        s.check(wait_for(fused, 1.5), 'cloud not fused while unreachable planning ran')
        burst = time.monotonic()
        send(2, (5., 1., 1.2))
        send(3)
        send(4, (-5., 0., 1.2))
        s.check(wait_for(lambda: any(m.command_sequence == 4 and m.goal_sequence == 4
                                     for _, m in states), 2),
                'last goal not acknowledged at an attempt boundary')
        latency = time.monotonic() - burst
        applied = [m.command_sequence for _, m in states if m.command_sequence in (2, 3, 4)]
        s.check(list(dict.fromkeys(applied)) == [2, 3, 4],
                f'commands not applied in publication order: {applied}')
        send(3)  # stale retransmission cannot erase goal 4
        d.cancel_pub.publish(Empty())  # legacy input must not mix into ordered mode
        time.sleep(.5)
        s.check(states[-1][1].command_sequence == 4 and states[-1][1].goal_sequence == 4,
                'stale or legacy cancel erased the newest goal')
        s.check(states[-1][1].state != 'WAIT_TARGET', 'newest goal was cancelled')
        send(5)
        s.check(wait_for(lambda: states[-1][1].command_sequence == 5
                         and states[-1][1].state == 'WAIT_TARGET', 3),
                'new cancel did not stop the goal')
        # The obstacle fused during failed planning must also constrain the
        # next published trajectory, not just appear in a visualization cloud.
        since = time.monotonic()
        send(6, (1.4, 1.4, 1.25))
        s.check(wait_for(lambda: any(not is_stationary(m) for m in d.bsplines_since(since)), 5),
                'no positive-control trajectory around the newly fused obstacle')
        candidates = [m for m in d.bsplines_since(since) if not is_stationary(m)]
        for msg in candidates:
            controls = np.array([[p.x, p.y, p.z] for p in msg.pos_pts])
            # Uniform cubic basis, independently sample the flown prefix.
            segments = len(controls) - 3
            for t in np.linspace(0, segments * 2 / 3, 300, endpoint=False):
                segment = int(t)
                u = t - segment
                weights = np.array([(1-u)**3, 3*u**3-6*u*u+4,
                                    -3*u**3+3*u*u+3*u+1, u**3]) / 6
                point = weights @ controls[segment:segment+4]
                if np.max(np.abs(point - probe[0])) < .29:
                    s.check(False, f'published trajectory crosses new inflated obstacle: {point}')
                    break
        # Invalid high sequences must not poison the receiver's watermark.
        pub.publish(PlannerCommand(sequence=100, command=255))
        send(100, (float('nan'), 0., 1.2))
        send(7)
        s.check(wait_for(lambda: states[-1][1].command_sequence == 7
                         and states[-1][1].state == 'WAIT_TARGET', 3),
                'invalid command advanced the sequence watermark')
        send(6, (1.4, 1.4, 1.25))
        time.sleep(.3)
        s.check(states[-1][1].command_sequence == 7 and states[-1][1].state == 'WAIT_TARGET',
                'stale goal revived a cancelled command')
        s.note(f'cloud fusion during failed planning: {fused()}; '
               f'map publications {len(maps)}; applied sequence {list(dict.fromkeys(applied))}; '
               f'ack latency {latency:.3f}s; late-obstacle trajectories checked {len(candidates)}')
    finally:
        node.destroy_publisher(pub)
        node.destroy_subscription(state_sub)
        node.destroy_subscription(map_sub)


if __name__ == '__main__':
    h = Harness()
    s = Scenario(h, 'ordered_commands')
    try:
        run(s)
    finally:
        s.close()
        h.shutdown()
    print('PASS' if not s.failures else 'FAIL', 'ordered_commands', s.notes, s.failures,
          flush=True)
    raise SystemExit(bool(s.failures))
