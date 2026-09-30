# Ordered commands and execution regression

SwarmDeck migration (legacy mode remains the default):

1. Set `fsm/ordered_commands: true` on `ego_planner_node`.
2. Replace both `/ROBOT/ego/goal` (`geometry_msgs/PoseStamped`) and
   `/ROBOT/ego/cancel` (`std_msgs/Empty`) publishers with **one** reliable,
   KeepAll `/ROBOT/ego/command` publisher of `traj_utils/PlannerCommand`.
   Ensure the adapter image installs/sources the updated `traj_utils` messages.
3. Set `command` to `GOAL=1` or `CANCEL=2`, and `sequence` to a positive,
   strictly increasing uint64 for **every** publication (including cancels).
   For goals, populate `goal` exactly as the old PoseStamped. The header stamp
   is not used for ordering. Use one serialized publisher/sequence allocator.
   Invalid command types/non-finite goals and duplicate/older sequences are
   ignored. Gaps are allowed; do not use multiple independent publishers.
4. Subscribe to `/ROBOT/ego/planning/command_state` (`traj_utils/CommandState`,
   reliable/transient-local, publisher depth 100). `command_sequence` is the
   last **applied** command, `goal_sequence` the last applied goal, and `state`
   is the FSM state atomically paired with those identities. Initial/legacy
   identities are zero. A cancel keeps the last goal identity and changes the
   command identity/state; completion also keeps the goal identity. A new
   goal is acknowledged even if the FSM state did not change. Do not count
   another goal's GEN_NEW_TRAJ (or INIT waiting for odometry) as this goal's
   planning progress. The old String `planning/fsm_state` remains unchanged.
5. Keep the relay's SOURCE_ADAPTER switch **before** publishing a cancel.
   Input receipt is independent of planning, but application/acknowledgment
   waits for the current optimization attempt to return. There is no hard
   real-time preemption of an individual optimizer call.

Sequence lifetime is the EGO process lifetime. An adapter-only restart must
retain its sequence allocator (greater than every previously issued command,
not merely the last acknowledged one), or restart EGO too. An EGO restart
resets identity to zero; the adapter must treat that as a new planner session.
Messages are volatile: explicitly resend a current goal with a new sequence
if the planner restarted. No legacy subscriptions exist in ordered mode, so
old Empty cancels cannot race migrated commands. In legacy mode both topics
still work, with KeepAll queues, but only *receipt* order is available: their
payloads do not encode a shared publication order.

## Execution ownership

Three mutually-exclusive callback groups run on three dedicated single-thread
executors: default/FSM/optimizer; map sensing/fusion/visualization; command and
odometry intake. Commands enter a mutex-protected FIFO, odometry a latest-value
inbox. Only the planning owner applies them, before its execution/safety
callbacks. Pending commands stop ordinary obsolete retries at attempt boundaries
and suppress ordinary obsolete candidate publication. Collision-safety replans
are never command-preemptible: a safe result is published even with commands
pending. If safety replanning fails for an imminent collision, the safety callback
publishes a stop immediately, before a queued goal can overwrite its state.
No whole-plan lock blocks intake or fusion. State heartbeats remain planning-owned
and can wait for an attempt; identities make late states unambiguous.

GridMap's mutable occupancy, rolling bounds, ground-column caches, and sensor
timeout state are protected by a recursive mutex. Each public query and map
update locks it. The final publication gate retains this lock over the same
upstream closest-2/3 occupancy check and existing full-tail flight-band check,
then publishes; rejected candidates restore the prior local trajectory. The
existing 50 ms in-flight collision/replanning timer is retained. It remains
serialized with the optimizer, as before; updates after publication are caught
on its next callback. Lidar timeout publishes an immediate stop too. Goals
received while the sensor timeout is active are applied and acknowledged with
the new identity, but remain pending in EMERGENCY_STOP until fresh sensing
resumes; this also applies in legacy mode. The final gate rejects a candidate
if the sensor times out during optimization. There is no change to inflation,
A*, optimizer costs, ceiling/recovery tolerances, or the upstream unchecked
obstacle-tail policy.

## Reproduce without a simulator

Build as in SwarmDeck's Dockerfile.drone, with `BUILD_TESTING=ON` for C++ tests:

```sh
source /opt/ros/jazzy/setup.bash
MAKEFLAGS=-j3 colcon build --parallel-workers 1 --packages-up-to ego_planner \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
source install/setup.bash
colcon test --packages-select plan_env ego_planner
colcon test-result --verbose
python3 src/planner/plan_manage/test/ordered_commands_check.py
python3 src/planner/plan_manage/test/swarmdeck_fsm_check.py
```

Use an isolated container (`--network none`, private ROS_DOMAIN_ID), not the
simulation stack. `ordered_commands_check.py` uses a real planner in a fused
closed box, waits for real failing A*, adds a new lidar voxel while planning,
and checks goal/cancel/goal application and stale/legacy cancel isolation.
`test_trajectory_publication_check` checks a late obstacle, atomic final
validation, and retention of the original unchecked-tail policy.
`test_safety_command_preemption` uses a Linux GNU/Clang test-only linker wrapper
around the optimizer result (and private access only on that test executable).
It exercises the real FSM/gate/rollback and ROS publications with deterministic
command or sensing interleavings, including immediate safety stops and sensor
recovery. No production injection hook is installed. It complements, rather than
replaces, the real-optimizer integration harness.
