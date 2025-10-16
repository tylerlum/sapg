import numpy as np
from scipy.spatial.transform import Rotation as R
import torch


def get_hammer_trajectory(object_init_state, device="cuda"):
    trajectory_states = []
    # first state is pick up state
    pick_up_state = object_init_state.copy()
    pick_up_state[3:7] = R.from_euler('y', 0, degrees=True).as_quat()
    pick_up_state[2] += 0.2

    # next state rotates 90 degrees around y axis
    rotate_90_state = pick_up_state.copy()
    rotate_90_state[3:7] = R.from_euler('y', 90, degrees=True).as_quat()

    # next state rotates 20 degrees around x axis
    swing_up_state = rotate_90_state.copy()
    swing_up_state[3:7] = R.from_euler('x', 20, degrees=True).as_quat()

    # next state rotates -40 degrees around x axis and swings hammer down
    swing_down_state = swing_up_state.copy()
    swing_down_state[3:7] = R.from_euler('x', -40, degrees=True).as_quat()
    swing_down_state[2] -= 0.1

    trajectory_states = [pick_up_state, rotate_90_state, swing_up_state, 
            swing_down_state, swing_up_state, swing_down_state]
    return torch.tensor(trajectory_states, dtype=torch.float32, device=device)

def get_screwdriver_trajectory(object_init_state, device="cuda"):
    trajectory_states = []
    # first state is pick up state
    pick_up_state = object_init_state.copy()
    pick_up_state[2] += 0.2

    # next state rotates 90 degrees around y axis
    rotate_90_state = pick_up_state.copy()
    rotate_90_state[3:7] = (R.from_quat(pick_up_state[3:7]) * R.from_euler('y', -90, degrees=True)).as_quat()

    # next state goes down
    go_down_state = rotate_90_state.copy()
    go_down_state[2] -= 0.1

    trajectory_states = [pick_up_state, rotate_90_state, go_down_state]
    # now we will sample screwing motion by rotating around x axis
    rotate_steps = np.linspace(0, -360, 20)
    num_cycles = 2
    for cycle in range(num_cycles):
        for step in rotate_steps:
            rotate_state = go_down_state.copy()
            rotate_state[3:7] = (R.from_quat(go_down_state[3:7]) * R.from_euler('x', step, degrees=True)).as_quat()
            trajectory_states.append(rotate_state)
    return torch.tensor(trajectory_states, dtype=torch.float32, device=device)
    
    
    
    