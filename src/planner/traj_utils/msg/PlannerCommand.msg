# One reliable publisher per planner lifetime; sequence starts at 1 and increases
# for every goal AND cancel. Retransmissions with sequence <= last received drop.
uint8 GOAL=1
uint8 CANCEL=2
uint64 sequence
uint8 command
# Used only for GOAL, in the same odometry frame as the legacy goal topic.
geometry_msgs/PoseStamped goal
