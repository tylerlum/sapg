"""DexToolBench Interactive Policy Demo
========================================
Single viser server with full IsaacGym policy evaluation.

Architecture:
  Main process  -- viser GUI + scene rendering (lightweight, no isaacgym)
  Subprocess    -- IsaacGym env + policy (sends state back via pipe)

Each "Load Environment" kills the old subprocess and spawns a fresh one,
sidestepping the fact that IsaacGym cannot cleanly reset within a process.

Usage:
    python dataset_viz.py [--port PORT]
"""

import argparse
import multiprocessing
import time
import traceback
from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf

# Pre-load the sidebar overview image as a numpy array (once, at import time)
_SIDEBAR_IMG_PATH = Path(__file__).resolve().parent / "dextoolbench_objects_sidebar.png"
_SIDEBAR_IMG = None
if _SIDEBAR_IMG_PATH.exists():
    from PIL import Image as _PILImage
    _SIDEBAR_IMG = np.asarray(_PILImage.open(_SIDEBAR_IMG_PATH).convert("RGB"))

# ═══════════════════════════════════════════════════════════════════
# Constants  (lightweight -- no isaacgym imports)
# ═══════════════════════════════════════════════════════════════════

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLE_Z = 0.38
Z_OFFSET = 0.03
TRAJ_SUFFIX = "_world_frame_min_z_0.6_downsampled_10"

# Default joint positions matching IsaacGym reset
# Arm: sharpa variant with startArmHigher=True
_ARM_DEFAULT = np.array([-1.571, 1.571, 0.0, 1.376, 0.0, 1.485, 1.308])
_ARM_DEFAULT[1] -= np.deg2rad(10)  # startArmHigher
_ARM_DEFAULT[3] += np.deg2rad(10)  # startArmHigher
DEFAULT_DOF_POS = np.zeros(29)
DEFAULT_DOF_POS[:7] = _ARM_DEFAULT

POLICY_DIR = Path(
    "/share/portal/kk837/sapg/train_dir/LATEST/FINETUNING_1x/"
    "NEW_FT_FixedSize_True_Force_True_Scale_2_2026-01-14_23-35-22"
)
_STEM = POLICY_DIR.name
CONFIG_PATH = POLICY_DIR / "runs" / f"00_{_STEM}" / "config.yaml"
CHECKPOINT_PATH = POLICY_DIR / "runs" / f"00_{_STEM}" / "best" / "model.pth"

# ── Dataset catalogue (from kushal_evals.py) ─────────────────────

CATEGORY_NAMES = {
    "Hammer": "hammer", "Spatula": "spatula", "Eraser": "eraser",
    "Screwdriver": "screwdriver", "Marker": "marker", "Brush": "brush",
}
OBJECTS = {
    "hammer":      {"Claw Hammer": "toy_hammer",           "Mallet Hammer": "mallet"},
    "spatula":     {"Black Spatula": "black_spatula",   "Spoon Spatula": "spoon_spatula"},
    "eraser":      {"Anvil Eraser": "anvil_eraser",     "Expo Eraser": "expo_eraser"},
    "screwdriver": {"Flat Screwdriver": "real_flat_screwdriver", "Red Screwdriver": "red_screwdriver"},
    "marker":      {"Sharpie": "sharpie_closed",        "Staples Marker": "staples_open"},
    "brush":       {"Red Brush": "red_brush",            "Anvil Brush": "anvil_brush"},
}
TRAJECTORIES = {
    "hammer":      {"Down Swing": "down_swing",          "Side Swing": "side_swing"},
    "spatula":     {"Serve Plate": "serve_plate",        "Flip Pancake": "flip_pancake"},
    "eraser":      {"Wipe Higher": "wipe_higher",        "Wipe Lower": "wipe_lower"},
    "screwdriver": {"Screw Top": "top",                  "Screw Side": "side"},
    "marker":      {"Write Smiley": "write_smiley",      "Write C": "write_c"},
    "brush":       {"Sweep Forward": "sweep_forward",    "Sweep Forward Right": "sweep_forward_right"},
}
TABLE_URDFS = {
    "hammer": "urdf/table_narrow_nail.urdf", "spatula": "urdf/table_narrow_bowl_plate.urdf",
    "eraser": "urdf/table_narrow_whiteboard.urdf", "screwdriver": "urdf/table_narrow.urdf",
    "marker": "urdf/table_narrow_whiteboard.urdf", "brush": "urdf/table_narrow.urdf",
}
OBJECT_URDFS = {
    "toy_hammer":            "assets/urdf/dex_tool_bench/hammer/toy_hammer/toy_hammer.urdf",
    "mallet":                "assets/urdf/dex_tool_bench/hammer/mallet/mallet.urdf",
    "black_spatula":         "assets/urdf/dex_tool_bench/spatula/black_spatula/spatula.urdf",
    "spoon_spatula":         "assets/urdf/dex_tool_bench/spatula/spoon_spatula/spoon_spatula.urdf",
    "anvil_eraser":          "assets/urdf/dex_tool_bench/eraser/anvil_eraser/anvil_eraser.urdf",
    "expo_eraser":           "assets/urdf/dex_tool_bench/eraser/expo_eraser/expo_eraser.urdf",
    "real_flat_screwdriver": "assets/urdf/dex_tool_bench/screwdriver/real_flat_screwdriver/real_flat_screwdriver.urdf",
    "red_screwdriver":       "assets/urdf/dex_tool_bench/screwdriver/red_screwdriver/red_screwdriver.urdf",
    "sharpie_closed":        "assets/urdf/dex_tool_bench/marker/sharpie_closed/sharpie_closed.urdf",
    "staples_open":          "assets/urdf/dex_tool_bench/marker/staples_open/staples_open.urdf",
    "red_brush":             "assets/urdf/dex_tool_bench/brush/red_brush/red_brush.urdf",
    "anvil_brush":           "assets/urdf/dex_tool_bench/brush/anvil_brush/anvil_brush.urdf",
}
CATEGORY_DESCRIPTIONS = {
    "hammer": "Swing a hammer to hit a nail.",
    "spatula": "Flip or serve food with a spatula.",
    "eraser": "Wipe a whiteboard with an eraser.",
    "screwdriver": "Drive a screw from the top or side.",
    "marker": "Write shapes on a whiteboard.",
    "brush": "Sweep debris forward across the table.",
}


def quat_xyzw_to_wxyz(q):
    return (q[3], q[0], q[1], q[2])


# ═══════════════════════════════════════════════════════════════════
# SUBPROCESS  -- IsaacGym simulation (all heavy imports stay here)
# ═══════════════════════════════════════════════════════════════════

def _sim_get_state(env, obs, joint_lower, joint_upper, n_act):
    """Extract visualisation state from the env."""
    obs_np = obs[0].cpu().numpy()
    joint_pos = 0.5 * (obs_np[:n_act] + 1.0) * (joint_upper - joint_lower) + joint_lower
    return (
        joint_pos,
        env.object_state[0, :7].cpu().numpy(),
        env.goal_pose[0].cpu().numpy(),
    )


def _sim_reset(env, n_act, device):
    import torch
    obs, _, _, _ = env.step(torch.zeros((env.num_envs, n_act), device=device))
    return obs["obs"]


def _sim_episode(conn, env, policy, joint_lower, joint_upper, n_act, device):
    """Run one episode, streaming state to the parent via *conn*."""
    import time, torch  # noqa: E401

    control_dt = 1.0 / 60.0

    policy.reset()
    obs = _sim_reset(env, n_act, device)

    step, done, paused = 0, False, False

    while not done:
        # Drain commands (non-blocking)
        while conn.poll(0):
            cmd = conn.recv()
            if cmd == "pause":
                paused = True
            elif cmd == "resume":
                paused = False
            elif cmd == "stop":
                conn.send(("stopped",))
                return obs

        if paused:
            time.sleep(0.05)
            continue

        t0 = time.time()

        state = _sim_get_state(env, obs, joint_lower, joint_upper, n_act)
        action = policy.get_normalized_action(obs, deterministic_actions=True)
        obs_dict, _, done_tensor, _ = env.step(action)
        obs = obs_dict["obs"]
        done = done_tensor[0].item()
        step += 1

        conn.send((
            "state",
            state,
            int(env.successes[0].item()),
            env.max_consecutive_successes,
            step,
        ))

        elapsed = time.time() - t0
        if (sleep := control_dt - elapsed) > 0:
            time.sleep(sleep)

    goal_pct = 100 * int(env.successes[0].item()) / env.max_consecutive_successes
    conn.send(("done", goal_pct, step))
    return obs


def sim_worker(conn, object_type, object_name, trajectory_name, table_urdf):
    """Child process entry-point.  Creates the env, then waits for commands."""
    # ── Heavy imports (only in the subprocess) ────────────────
    from isaacgym import gymapi  # noqa: F401 isort:skip
    import json, torch  # noqa: E401
    from isaacgymenvs.utils.utils import get_repo_root_dir
    from sim2real.rl_player import RlPlayer
    from sim2sim.isaac_sim.isaac_env import create_env

    n_act = 29
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Load trajectory
        traj_path = (
            get_repo_root_dir() / "dex_tool_bench" / "evaluation_trajectories"
            / object_type / object_name / f"{trajectory_name}.json"
        )
        with open(traj_path) as f:
            traj_data = json.load(f)
        traj_data["start_pose"][2] += Z_OFFSET

        # Create environment
        env = create_env(
            config_path=str(CONFIG_PATH), headless=True, device=device,
            overrides={
                "task.env.resetPositionNoiseX": 0.0,
                "task.env.resetPositionNoiseY": 0.0,
                "task.env.resetPositionNoiseZ": 0.0,
                "task.env.resetRotationNoise": 0.0,
                "task.env.resetDofPosRandomIntervalFingers": 0.0,
                "task.env.resetDofPosRandomIntervalArm": 0.0,
                "task.env.resetDofVelRandomInterval": 0.0,
                "task.env.object_type": object_name,
                "task.env.randomizeObjectRotation": False,
                "task.env.numEnvs": 1,
                "task.env.envSpacing": 0.4,
                "task.env.tableResetZRange": 0.0,
                "task.env.capture_video": False,
                "task.env.use_fixed_set_of_goal_states": True,
                "task.env.fixedGoalStates": traj_data["goals"],
                "task.env.objectStartPose": traj_data["start_pose"],
                "task.env.useActionDelay": False,
                "task.env.useObsDelay": False,
                "task.env.useObjectStateDelayNoise": False,
                "task.env.resetWhenDropped": False,
                "task.env.armMovingAverage": 0.1,
                "task.env.evalSuccessTolerance": 0.01,
                "task.env.successSteps": 1,
                "task.env.asset.table": str(table_urdf),
                "task.env.tableResetZ": TABLE_Z,
                "task.env.fixedSizeKeypointReward": True,
                "task.env.startArmHigher": True,
                "task.env.objectScaleNoiseMultiplierRange": [1.0, 1.0],
                "task.env.forceScale": 0.0, "task.env.torqueScale": 0.0,
                "task.env.linVelImpulseScale": 0.0, "task.env.angVelImpulseScale": 0.0,
                "task.env.forceOnlyWhenLifted": True, "task.env.torqueOnlyWhenLifted": True,
                "task.env.linVelImpulseOnlyWhenLifted": True, "task.env.angVelImpulseOnlyWhenLifted": True,
                "task.env.forceProbRange": [0.0001, 0.0001],
                "task.env.torqueProbRange": [0.0001, 0.0001],
                "task.env.linVelImpulseProbRange": [0.0001, 0.0001],
                "task.env.angVelImpulseProbRange": [0.0001, 0.0001],
            },
        )

        joint_lower = env.arm_hand_dof_lower_limits[:n_act].cpu().numpy()
        joint_upper = env.arm_hand_dof_upper_limits[:n_act].cpu().numpy()

        # Load policy
        env.set_env_state(torch.load(CHECKPOINT_PATH)[0]["env_state"])
        policy = RlPlayer(140, n_act, CONFIG_PATH, CHECKPOINT_PATH, device, env.num_envs)

        # Initial reset
        obs = _sim_reset(env, n_act, device)
        init_state = _sim_get_state(env, obs, joint_lower, joint_upper, n_act)

        conn.send(("ready", init_state))

        # ── Command loop ─────────────────────────────────────
        while True:
            cmd = conn.recv()
            if cmd == "run":
                obs = _sim_episode(
                    conn, env, policy, joint_lower, joint_upper, n_act, device,
                )
            elif cmd == "quit":
                break

    except Exception as exc:
        conn.send(("error", f"{exc}\n{traceback.format_exc()}"))

    conn.close()


# ═══════════════════════════════════════════════════════════════════
# MAIN PROCESS  -- single viser server with all GUI + rendering
# ═══════════════════════════════════════════════════════════════════

class InteractiveDemo:

    def __init__(self, port: int = 8080):
        self.port = port
        self.server = viser.ViserServer(host="0.0.0.0", port=port)

        # Subprocess
        self._proc: multiprocessing.Process | None = None
        self._conn: multiprocessing.connection.Connection | None = None
        self._env_ready = False
        self._episode_running = False
        self._is_paused = False

        # Pending config (set in _load_env, consumed in _handle_ready)
        self._pending_obj_key: str = ""
        self._pending_table_urdf: str = ""

        # Stats
        self.ep_count = 0
        self.ep_goals: list[float] = []
        self.ep_lengths: list[int] = []

        # Scene handles
        self.robot: ViserUrdf | None = None
        self._dyn: list = []
        self._obj_frame = None
        self._goal_frame = None

        self._build_gui()
        self._setup_static_scene()

    # ── GUI ────────────────────────────────────────────────────

    def _build_gui(self):
        self.server.gui.add_markdown(
            "# DexToolBench\n### Interactive Policy Demo"
        )

        if _SIDEBAR_IMG is not None:
            with self.server.gui.add_folder(
                "DexToolBench Objects", expand_by_default=True,
            ):
                self.server.gui.add_image(
                    _SIDEBAR_IMG,
                    label="Tool objects in the benchmark",
                    format="jpeg",
                )

        _PH = "-- Select --"
        with self.server.gui.add_folder("Dataset Selection", expand_by_default=True):
            cats = [_PH] + list(CATEGORY_NAMES.keys())
            self._dd_cat = self.server.gui.add_dropdown(
                "Tool Category", options=cats, initial_value=_PH,
            )
            self._dd_obj = self.server.gui.add_dropdown(
                "Object Instance", options=[_PH], initial_value=_PH,
            )
            self._dd_traj = self.server.gui.add_dropdown(
                "Trajectory", options=[_PH], initial_value=_PH,
            )
            self._md_desc = self.server.gui.add_markdown(
                "*Select a tool category to begin.*"
            )
            self._btn_load = self.server.gui.add_button("Load Environment")
            self._btn_load.on_click(lambda _: self._load_env())
            self._dd_cat.on_update(lambda _: self._on_cat_change())

        with self.server.gui.add_folder("Episode Controls", expand_by_default=True):
            self._btn_run = self.server.gui.add_button("Run Episode")
            self._btn_run.on_click(lambda _: self._cmd_run())
            self._btn_pause = self.server.gui.add_button("Pause")
            self._btn_pause.on_click(lambda _: self._cmd_pause())
            self._btn_stop = self.server.gui.add_button("Stop Episode")
            self._btn_stop.on_click(lambda _: self._cmd_stop())

        with self.server.gui.add_folder("Status", expand_by_default=True):
            self._md_task = self.server.gui.add_markdown("**Task:** --")
            self._md_prog = self.server.gui.add_markdown("**Progress:** --")
            self._md_stats = self.server.gui.add_markdown("**Stats:** No episodes yet")
            self._md_obj = self.server.gui.add_markdown("**Object Pos:** --")
            self._md_status = self.server.gui.add_markdown("**Status:** Ready")

    # ── Static scene ───────────────────────────────────────────

    def _setup_static_scene(self):
        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            client.camera.position = (0.0, -1.0, 1.0)
            client.camera.look_at = (0.0, 0.0, 0.5)

        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

        robot_urdf = (
            REPO_ROOT / "assets" / "urdf" / "kuka_allegro_description"
            / "iiwa14_left_sharpa_adjusted_restricted.urdf"
        )
        self.server.scene.add_frame(
            "/robot", position=(0, 0.8, 0), wxyz=(1, 0, 0, 0), show_axes=False,
        )
        self.robot = ViserUrdf(self.server, robot_urdf, root_node_name="/robot")
        self.robot.update_cfg(DEFAULT_DOF_POS)

        # Show a default table before any environment is loaded
        self._setup_table("urdf/table_narrow.urdf")

    # ── Cascading dropdown ─────────────────────────────────────

    def _on_cat_change(self):
        cat = self._dd_cat.value
        if cat not in CATEGORY_NAMES:
            return
        ck = CATEGORY_NAMES[cat]
        self._dd_obj.options = list(OBJECTS[ck].keys())
        self._dd_obj.value = list(OBJECTS[ck].keys())[0]
        self._dd_traj.options = list(TRAJECTORIES[ck].keys())
        self._dd_traj.value = list(TRAJECTORIES[ck].keys())[0]
        self._md_desc.content = f"*{CATEGORY_DESCRIPTIONS[ck]}*"

    # ── Dynamic scene (rebuilt per config) ─────────────────────

    def _clear_dynamic(self):
        for h in self._dyn:
            try:
                h.remove()
            except Exception:
                pass
        self._dyn.clear()
        self._obj_frame = self._goal_frame = None

    def _add_box(self, name, dimensions, position, color, opacity=None):
        """Add a coloured box to the viser scene and track it in _dyn."""
        kwargs = dict(color=color, dimensions=dimensions, position=position, side="double")
        if opacity is not None:
            kwargs["opacity"] = opacity
        h = self.server.scene.add_box(name, **kwargs)
        self._dyn.append(h)
        return h

    def _setup_table(self, table_urdf_rel):
        """Load the table immediately (we know which one before IsaacGym is ready)."""
        self._clear_dynamic()
        t = self.server.scene.add_frame(
            "/table", position=(0, 0, TABLE_Z), wxyz=(1, 0, 0, 0), show_axes=False,
        )
        self._dyn.append(t)

        if "bowl_plate" in table_urdf_rel:
            # Bowl / plate use actual .obj meshes -- keep ViserUrdf
            ViserUrdf(self.server, REPO_ROOT / "assets" / table_urdf_rel,
                      root_node_name="/table")
        else:
            # Wooden table surface
            self._add_box(
                "/table/wood", (0.475, 0.4, 0.3), (0, 0, 0),
                color=(180, 130, 70), opacity=0.9,
            )

            if "nail" in table_urdf_rel:
                # Metallic silver nail
                self._add_box(
                    "/table/nail", (0.03, 0.03, 0.04), (-0.21, 0.06, 0.165),
                    color=(170, 175, 180),
                )
            elif "whiteboard" in table_urdf_rel:
                bx = 0.345  # board center x
                bw, bh = 0.40, 0.40  # board width (y) and height (z)
                bc_y, bc_z = 0.0, 0.35  # board center y, z
                fw = 0.03  # frame strip width
                fd = 0.03  # frame depth (x thickness)
                # Green chalkboard surface
                self._add_box(
                    "/table/wb_surface", (0.02, bw, bh), (bx, bc_y, bc_z),
                    color=(60, 130, 60),
                )
                # Wooden frame: 4 border strips
                # Top strip
                self._add_box(
                    "/table/wb_ft", (fd, bw + 2 * fw, fw),
                    (bx, bc_y, bc_z + bh / 2 + fw / 2),
                    color=(140, 90, 45),
                )
                # Bottom strip
                self._add_box(
                    "/table/wb_fb", (fd, bw + 2 * fw, fw),
                    (bx, bc_y, bc_z - bh / 2 - fw / 2),
                    color=(140, 90, 45),
                )
                # Left strip
                self._add_box(
                    "/table/wb_fl", (fd, fw, bh),
                    (bx, bc_y - bw / 2 - fw / 2, bc_z),
                    color=(140, 90, 45),
                )
                # Right strip
                self._add_box(
                    "/table/wb_fr", (fd, fw, bh),
                    (bx, bc_y + bw / 2 + fw / 2, bc_z),
                    color=(140, 90, 45),
                )

        # Reset robot to default pose while we wait
        self.robot.update_cfg(DEFAULT_DOF_POS)

    def _setup_object_goal(self, obj_key):
        """Add the object + goal URDFs (called once IsaacGym reports ready)."""
        obj_urdf = REPO_ROOT / OBJECT_URDFS[obj_key]

        self._obj_frame = self.server.scene.add_frame(
            "/object", show_axes=True, axes_length=0.1, axes_radius=0.001,
        )
        self._dyn.append(self._obj_frame)
        ViserUrdf(self.server, obj_urdf, root_node_name="/object")

        self._goal_frame = self.server.scene.add_frame(
            "/goal", show_axes=True, axes_length=0.1, axes_radius=0.001,
        )
        self._dyn.append(self._goal_frame)
        ViserUrdf(self.server, obj_urdf, root_node_name="/goal",
                  mesh_color_override=(0, 255, 0, 0.5))

    # ── Subprocess management ──────────────────────────────────

    def _kill_subprocess(self):
        if self._conn is not None:
            try:
                self._conn.send("quit")
            except (BrokenPipeError, OSError):
                pass
            self._conn.close()
            self._conn = None
        if self._proc is not None:
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join()
            self._proc = None
        self._env_ready = False
        self._episode_running = False
        self._is_paused = False

    def _load_env(self):
        if self._dd_cat.value not in CATEGORY_NAMES:
            self._md_status.content = "**Status:** Please select a tool category first."
            return
        self._kill_subprocess()

        cat_key = CATEGORY_NAMES[self._dd_cat.value]
        obj_key = OBJECTS[cat_key][self._dd_obj.value]
        traj_key = TRAJECTORIES[cat_key][self._dd_traj.value]
        traj_name = f"{traj_key}{TRAJ_SUFFIX}"
        table_urdf = TABLE_URDFS[cat_key]

        self._pending_obj_key = obj_key

        # Show table + default robot pose immediately while IsaacGym loads
        self._setup_table(table_urdf)

        label = f"{self._dd_cat.value} / {self._dd_obj.value} / {self._dd_traj.value}"
        self._md_status.content = f"**Status:** Loading *{label}* ..."
        self._md_task.content = f"**Task:** {label}"

        # Reset stats
        self.ep_count = 0
        self.ep_goals.clear()
        self.ep_lengths.clear()
        self._md_stats.content = "**Stats:** No episodes yet"

        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self._conn = parent_conn
        self._proc = ctx.Process(
            target=sim_worker,
            args=(child_conn, cat_key, obj_key, traj_name, table_urdf),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()
        print(f"[launcher] Spawned subprocess pid={self._proc.pid}")

    # ── Commands to subprocess ─────────────────────────────────

    def _send(self, msg):
        if self._conn is not None:
            try:
                self._conn.send(msg)
            except (BrokenPipeError, OSError):
                pass

    def _cmd_run(self):
        if not self._env_ready:
            self._md_status.content = "**Status:** Load an environment first."
            return
        if self._episode_running:
            return
        self._episode_running = True
        self._is_paused = False
        self._btn_pause.name = "Pause"
        self._md_status.content = "**Status:** Running episode..."
        self._send("run")

    def _cmd_pause(self):
        if not self._episode_running:
            return
        self._is_paused = not self._is_paused
        self._send("pause" if self._is_paused else "resume")
        self._btn_pause.name = "Resume" if self._is_paused else "Pause"

    def _cmd_stop(self):
        if self._episode_running:
            self._send("stop")

    # ── Scene update ───────────────────────────────────────────

    def _update_viz(self, state_tuple):
        joint_pos, obj_pose, goal_pose = state_tuple
        self.robot.update_cfg(joint_pos)
        if self._obj_frame is not None:
            self._obj_frame.position = tuple(obj_pose[:3])
            self._obj_frame.wxyz = quat_xyzw_to_wxyz(obj_pose[3:7])
        if self._goal_frame is not None:
            self._goal_frame.position = tuple(goal_pose[:3])
            self._goal_frame.wxyz = quat_xyzw_to_wxyz(goal_pose[3:7])

    # ── Message handling ───────────────────────────────────────

    def _handle(self, msg):
        tag = msg[0]

        if tag == "ready":
            init_state = msg[1]
            self._setup_object_goal(self._pending_obj_key)
            self._update_viz(init_state)
            self._env_ready = True
            self._md_status.content = "**Status:** Ready -- click **Run Episode**"
            print("[launcher] Environment ready")

        elif tag == "state":
            state, successes, max_succ, step = msg[1], msg[2], msg[3], msg[4]
            self._update_viz(state)
            pct = 100 * successes / max_succ if max_succ > 0 else 0
            self._md_prog.content = (
                f"**Time:** {step / 60.0:.1f}s &nbsp;|&nbsp; "
                f"**Goal:** {successes}/{max_succ} ({pct:.0f}%)"
            )
            obj_pos = state[1][:3]
            self._md_obj.content = (
                f"**Object Pos:** {obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}"
            )

        elif tag == "done":
            goal_pct, steps = msg[1], msg[2]
            self._episode_running = False
            self.ep_goals.append(goal_pct)
            self.ep_lengths.append(steps)
            self.ep_count += 1
            avg_g = np.mean(self.ep_goals)
            avg_t = np.mean(self.ep_lengths) / 60.0
            self._md_stats.content = (
                f"**Episodes:** {self.ep_count} &nbsp;|&nbsp; "
                f"**Avg Goal:** {avg_g:.1f}% &nbsp;|&nbsp; "
                f"**Avg Time:** {avg_t:.1f}s"
            )
            self._md_status.content = (
                f"**Status:** Done -- {steps / 60.0:.1f}s, {goal_pct:.0f}% goals"
            )
            print(f"[launcher] Episode done: {goal_pct:.0f}% goals in {steps / 60.0:.1f}s")

        elif tag == "stopped":
            self._episode_running = False
            self._md_status.content = "**Status:** Episode stopped."

        elif tag == "error":
            self._env_ready = False
            self._episode_running = False
            self._md_status.content = f"**Status:** Error -- {msg[1][:200]}"
            print(f"[launcher] Subprocess error:\n{msg[1]}")

    def _poll(self):
        """Drain all pending messages from the subprocess pipe."""
        if self._conn is None:
            return
        try:
            while self._conn.poll(0):
                self._handle(self._conn.recv())
        except (EOFError, ConnectionResetError, OSError):
            self._conn = None
            if self._proc is not None and not self._proc.is_alive():
                self._md_status.content = "**Status:** Subprocess exited unexpectedly."
                self._proc = None
                self._env_ready = False
                self._episode_running = False

    # ── Main loop ──────────────────────────────────────────────

    def run(self):
        print()
        print("  +-------------------------------------------------+")
        print("  |     DexToolBench Interactive Policy Demo         |")
        print(f"  |     http://localhost:{self.port:<26}|")
        print("  +-------------------------------------------------+")
        print()

        try:
            while True:
                self._poll()
                time.sleep(1.0 / 120.0)
        except KeyboardInterrupt:
            print("\n[launcher] Shutting down...")
            self._kill_subprocess()


# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DexToolBench Interactive Policy Demo",
    )
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    InteractiveDemo(port=args.port).run()
