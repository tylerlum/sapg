import isaacgym
import isaacgymenvs
import torch
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="AllegroKuka")
parser.add_argument("--num_envs", type=int, default=10)
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()
if args.device == "cuda":
	device = 'cuda:0'
	num_envs = args.num_envs
	envs = isaacgymenvs.make(
		seed=0, 
		task=args.task, 
		num_envs=num_envs, 
		sim_device=device,
		rl_device=device,
		graphics_device_id=0,
	)
else:
	device = 'cpu'
	num_envs = args.num_envs
	envs = isaacgymenvs.make(
		seed=0, 
		task=args.task, 
		num_envs=num_envs, 
		sim_device=device,
		rl_device=device,
		graphics_device_id=0,
	)
print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)
obs = envs.reset()
# breakpoint()
for _ in range(20000):
	random_actions = 2.0 * torch.rand((num_envs,) + envs.action_space.shape, device = device) - 1.0
	# breakpoint()
	envs.step(random_actions*0)
	# breakpoint()
	print("step")
