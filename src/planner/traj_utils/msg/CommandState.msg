# Applied (not merely received) command and goal, atomically paired with state.
# Zero means no ordered command/goal has been applied. goal_sequence retains
# the last goal's identity after cancellation/completion; inspect state too.
uint64 command_sequence
uint64 goal_sequence
string state
