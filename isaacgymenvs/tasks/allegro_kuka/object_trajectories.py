import numpy as np
from scipy.spatial.transform import Rotation as R
import torch


def get_hammer_trajectory(object_init_state, device="cuda"):
    trajectory_states = []
    # first state is pick up state
    pick_up_state = object_init_state.copy()
    pick_up_state[3:7] = R.from_euler('y', 0, degrees=True).as_quat()
    pick_up_state[2] += 0.2

    # next state rotates -90 degrees around x axis wrt last state
    rotate_90_state = pick_up_state.copy()
    rotate_90_state[3:7] = (R.from_quat(pick_up_state[3:7]) * R.from_euler('x', -90, degrees=True)).as_quat()

    trajectory_states = [pick_up_state, rotate_90_state]
    # next state rotates 20 degrees around z axis wrt last state
    swing_up_state = rotate_90_state.copy()
    swing_up_state[3:7] = (R.from_quat(rotate_90_state[3:7]) * R.from_euler('z', -20, degrees=True)).as_quat()

    # next state rotates -40 degrees around x axis and swings hammer down wrt last state
    swing_down_state = swing_up_state.copy()
    swing_down_state[3:7] = (R.from_quat(swing_up_state[3:7]) * R.from_euler('z', 60, degrees=True)).as_quat()
    swing_down_state[2] -= 0.15

    num_swings = 4
    for _ in range(num_swings):
        trajectory_states.append(swing_up_state)
        trajectory_states.append(swing_down_state)

    # trajectory_states = [pick_up_state, rotate_90_state, swing_up_state, 
    #         swing_down_state, swing_up_state, swing_down_state]
    return torch.tensor(trajectory_states, dtype=torch.float32, device=device)

def get_hairbrush_trajectory(object_init_state, device="cuda"):
    trajectory_states = []
    # first state is pick up state
    pick_up_state = object_init_state.copy()
    pick_up_state[3:7] = R.from_euler('y', 0, degrees=True).as_quat()
    pick_up_state[2] += 0.2

    # next state rotates -90 degrees around x axis wrt last state
    rotate_180_state = pick_up_state.copy()
    rotate_180_state[3:7] = (R.from_quat(pick_up_state[3:7]) * R.from_euler('x', -180, degrees=True)).as_quat()

    trajectory_states = [pick_up_state, rotate_180_state]

    middle_state = rotate_180_state.copy()

    # next state moves forward +y by 0.1
    move_forward_state = middle_state.copy()
    move_forward_state[1] += 0.1

    move_half_forward_state = middle_state.copy()
    move_half_forward_state[1] += 0.05

    # next state moves backward -y by 0.1
    move_backward_state = middle_state.copy()
    move_backward_state[1] -= 0.1

    move_half_backward_state = middle_state.copy()
    move_half_backward_state[1] -= 0.05

    num_moves = 4
    for _ in range(num_moves):
        trajectory_states.append(move_half_forward_state)
        trajectory_states.append(move_forward_state)
        trajectory_states.append(move_half_forward_state)
        trajectory_states.append(middle_state)
        trajectory_states.append(move_half_backward_state)
        trajectory_states.append(move_backward_state)
        trajectory_states.append(move_half_backward_state)
        trajectory_states.append(middle_state)

    return torch.tensor(trajectory_states, dtype=torch.float32, device=device)

def get_screwdriver_trajectory(object_init_state, device="cuda"):
    trajectory_states = []
    # first state is pick up state
    pick_up_state = object_init_state.copy()
    pick_up_state[3:7] = R.from_euler('y', 0, degrees=True).as_quat()
    pick_up_state[2] += 0.2

    # next state rotates 90 degrees around y axis
    rotate_90_state = pick_up_state.copy()
    rotate_90_state[3:7] = (R.from_quat(pick_up_state[3:7]) * R.from_euler('y', 90, degrees=True)).as_quat()

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

# def get_marker_trajectory(object_init_state, device="cuda"):
#     trajectory_states = []
#     # first state is pick up state
#     pick_up_state = object_init_state.copy()
#     pick_up_state[3:7] = R.from_euler('y', 0, degrees=True).as_quat()
#     pick_up_state[2] += 0.2

#     # next state rotates 90 degrees around y axis
#     rotate_90_state = pick_up_state.copy()
#     rotate_90_state[3:7] = (R.from_quat(pick_up_state[3:7]) * R.from_euler('y', -90, degrees=True)).as_quat()

#     # next state goes down
#     go_down_state = rotate_90_state.copy()
#     go_down_state[2] -= 0.1

#     trajectory_states = [pick_up_state, rotate_90_state, go_down_state]
#     # now we will sample screwing motion by rotating around x axis
#     rotate_steps = np.linspace(0, -360, 20)
#     num_cycles = 2
#     for cycle in range(num_cycles):
#         for step in rotate_steps:
#             rotate_state = go_down_state.copy()
#             rotate_state[3:7] = (R.from_quat(go_down_state[3:7]) * R.from_euler('x', step, degrees=True)).as_quat()
#             trajectory_states.append(rotate_state)
#     return torch.tensor(trajectory_states, dtype=torch.float32, device=device)
def get_marker_trajectory(object_init_state, device="cuda"):
    trajectory_states = []
    # first state is pick up state
    pick_up_state = object_init_state.copy()
    pick_up_state[3:7] = R.from_euler('y', 0, degrees=True).as_quat()
    pick_up_state[2] += 0.2

    # next state rotates 90 degrees around y axis
    rotate_90_state = pick_up_state.copy()
    rotate_90_state[3:7] = (R.from_quat(pick_up_state[3:7]) * R.from_euler('y', -90, degrees=True)).as_quat()

    # next state goes right
    go_right_state = rotate_90_state.copy()
    go_right_state[0] += 0.2

    # next state goes down
    go_down_state = go_right_state.copy()
    go_down_state[2] -= 0.05

    # next state goes down
    go_down_down_state = go_down_state.copy()
    go_down_down_state[2] -= 0.05

    # next state goes down
    go_down_down_down_state = go_down_down_state.copy()
    go_down_down_down_state[2] -= 0.05

    # next state goes down
    go_down_down_down_down_state = go_down_down_down_state.copy()
    go_down_down_down_down_state[2] -= 0.05

    trajectory_states = [pick_up_state, rotate_90_state, go_right_state, go_down_state, go_down_down_state, go_down_down_down_state, go_down_down_down_down_state]
    # 
    return torch.tensor(trajectory_states, dtype=torch.float32, device=device)

def get_eraser_trajectory(object_init_state, device="cuda"):
    trajectory_states = []
    # first state is pick up state
    pick_up_state = object_init_state.copy()
    pick_up_state[3:7] = R.from_euler('y', 0, degrees=True).as_quat()
    pick_up_state[2] += 0.2

    # next state rotates 90 degrees around z axis wrt last state
    rotate_90_state = pick_up_state.copy()
    rotate_90_state[3:7] = (R.from_quat(pick_up_state[3:7]) * R.from_euler('z', 90, degrees=True)).as_quat()

    # next state goes right
    go_right_state = rotate_90_state.copy()
    go_right_state[0] += 0.2

    trajectory_states = [pick_up_state, rotate_90_state, go_right_state]

    num_strokes = 4
    # define go up state and go down state
    go_up_state = go_right_state.copy()
    go_up_state[2] += 0.05
    go_down_state = go_up_state.copy()
    go_down_state[2] -= 0.1

    for _ in range(num_strokes):
        trajectory_states.append(go_up_state)
        trajectory_states.append(go_down_state)
    
    return torch.tensor(trajectory_states, dtype=torch.float32, device=device)

def get_phone_trajectory(object_init_state, device="cuda"):
    trajectory_states = []
    # first state is pick up state and rotation 180 degrees around x axis
    pick_up_state = object_init_state.copy()
    pick_up_state[3:7] = R.from_euler('x', 180, degrees=True).as_quat()
    pick_up_state[2] += 0.1

    # next state rotates 90 degrees around y axis
    rotate_90_state = pick_up_state.copy()
    rotate_90_state[3:7] = (R.from_quat(pick_up_state[3:7]) * R.from_euler('z', 90, degrees=True)).as_quat()

    # next state rotates 90 degrees around x axis
    rotate_90_again_state = rotate_90_state.copy()
    rotate_90_again_state[3:7] = (R.from_quat(rotate_90_state[3:7]) * R.from_euler('y', 90, degrees=True)).as_quat()

    # next state reduces y position by 0.05
    reduce_y_state = rotate_90_again_state.copy()
    reduce_y_state[1] += 0.05

    # next state reduces y position by 0.05
    reduce_y_again_state = reduce_y_state.copy()
    reduce_y_again_state[1] += 0.05

    # next state reduces y position by 0.05
    reduce_y_again_again_state = reduce_y_again_state.copy()
    reduce_y_again_again_state[1] += 0.05

    trajectory_states = [pick_up_state, rotate_90_again_state, reduce_y_state, reduce_y_again_state, reduce_y_again_again_state]
    return torch.tensor(trajectory_states, dtype=torch.float32, device=device)
    
    
    