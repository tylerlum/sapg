from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
from jax.typing import ArrayLike


def _solve_trajopt_core(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_link_index: int,
    start_position: jax.Array,
    start_wxyz: jax.Array,
    end_position: jax.Array,
    end_wxyz: jax.Array,
    timesteps: int,
    dt: float,
) -> jax.Array:
    """Core optimization logic. Pure JAX, no strings, no numpy checks."""
    
    # 1. Solve IK for the start and end poses.
    start_cfg, end_cfg = solve_iks_with_collision(
        robot=robot,
        coll=robot_coll,
        world_coll_list=world_coll,
        target_link_index=target_link_index,
        target_position_0=start_position,
        target_wxyz_0=start_wxyz,
        target_position_1=end_position,
        target_wxyz_1=end_wxyz,
    )

    # 2. Initialize trajectory
    init_traj = jnp.linspace(start_cfg, end_cfg, timesteps)

    # 3. Optimize
    traj_vars = robot.joint_var_cls(jnp.arange(timesteps))

    # Add batch dims for internal cost evaluation
    robot_b = jax.tree.map(lambda x: x[None], robot)
    robot_coll_b = jax.tree.map(lambda x: x[None], robot_coll)

    factors: list[jaxls.Cost] = [
        pk.costs.rest_cost(
            traj_vars,
            traj_vars.default_factory()[None],
            jnp.array([0.01])[None],
        ),
        pk.costs.limit_cost(
            robot_b,
            traj_vars,
            jnp.array([100.0])[None],
        ),
    ]

    # Collision avoidance closure
    def compute_world_coll_residual(vals, r, rc, wc, prev, curr):
        coll = rc.get_swept_capsules(r, vals[prev], vals[curr])
        dist = pk.collision.collide(coll.reshape((-1, 1)), wc.reshape((1, -1)))
        colldist = pk.collision.colldist_from_sdf(dist, 0.1)
        return (colldist * 20.0).flatten()

    for world_coll_obj in world_coll:
        factors.append(
            jaxls.Cost(
                compute_world_coll_residual,
                (
                    robot_b,
                    robot_coll_b,
                    jax.tree.map(lambda x: x[None], world_coll_obj),
                    robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                    robot.joint_var_cls(jnp.arange(1, timesteps)),
                ),
                name="World Collision (sweep)",
            )
        )

    # Constraints and Smoothing
    factors.extend([
        jaxls.Cost(
            lambda vals, var: ((vals[var] - start_cfg) * 100.0).flatten(),
            (robot.joint_var_cls(jnp.arange(0, 2)),),
            name="start_pose_constraint",
        ),
        jaxls.Cost(
            lambda vals, var: ((vals[var] - end_cfg) * 100.0).flatten(),
            (robot.joint_var_cls(jnp.arange(timesteps - 2, timesteps)),),
            name="end_pose_constraint",
        ),
    ])

    factors.extend([
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            jnp.array([0.1])[None],
        ),
        pk.costs.five_point_velocity_cost(
            robot_b,
            robot.joint_var_cls(jnp.arange(4, timesteps)),
            robot.joint_var_cls(jnp.arange(3, timesteps - 1)),
            robot.joint_var_cls(jnp.arange(1, timesteps - 3)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 4)),
            dt,
            jnp.array([10.0])[None],
        ),
        pk.costs.five_point_acceleration_cost(
            robot.joint_var_cls(jnp.arange(2, timesteps - 2)),
            robot.joint_var_cls(jnp.arange(4, timesteps)),
            robot.joint_var_cls(jnp.arange(3, timesteps - 1)),
            robot.joint_var_cls(jnp.arange(1, timesteps - 3)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 4)),
            dt,
            jnp.array([0.1])[None],
        ),
        pk.costs.five_point_jerk_cost(
            robot.joint_var_cls(jnp.arange(6, timesteps)),
            robot.joint_var_cls(jnp.arange(5, timesteps - 1)),
            robot.joint_var_cls(jnp.arange(4, timesteps - 2)),
            robot.joint_var_cls(jnp.arange(2, timesteps - 4)),
            robot.joint_var_cls(jnp.arange(1, timesteps - 5)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 6)),
            dt,
            jnp.array([0.1])[None],
        ),
    ])

    solution = (
        jaxls.LeastSquaresProblem(factors, [traj_vars])
        .analyze()
        .solve(initial_vals=jaxls.VarValues.make((traj_vars.with_value(init_traj),)))
    )
    return solution[traj_vars]


# @jax.jit
# def solve_trajopt_batched(
#     robot: pk.Robot,
#     robot_coll: pk.collision.RobotCollision,
#     world_coll: Sequence[pk.collision.CollGeom],
#     target_link_index: int,
#     start_positions: jax.Array,  # (N, 3)
#     start_wxyzs: jax.Array,      # (N, 4)
#     end_positions: jax.Array,    # (N, 3)
#     end_wxyzs: jax.Array,        # (N, 4)
#     timesteps: int,
#     dt: float,
# ) -> jax.Array:  # Returns (N, timesteps, dof)
#     return jax.vmap(
#         _solve_trajopt_core,
#         in_axes=(None, None, None, None, 0, 0, 0, 0, None, None),
#     )(
#         robot, robot_coll, world_coll, target_link_index,
#         start_positions, start_wxyzs, end_positions, end_wxyzs,
#         timesteps, dt
#     )
# CHANGE HERE: Add static_argnames to the jit decorator
# @jax.jit(static_argnames=["timesteps", "dt"])
# def solve_trajopt_batched(
#     robot: pk.Robot,
#     robot_coll: pk.collision.RobotCollision,
#     world_coll: Sequence[pk.collision.CollGeom],
#     target_link_index: int,
#     start_positions: jax.Array,
#     start_wxyzs: jax.Array,
#     end_positions: jax.Array,
#     end_wxyzs: jax.Array,
#     timesteps: int,
#     dt: float,
# ) -> jax.Array:
#     """
#     Batched solver. 
#     'timesteps' and 'dt' must be static because they determine array shapes 
#     (e.g. inside jnp.linspace and for loop bounds).
#     """
#     return jax.vmap(
#         _solve_trajopt_core,
#         # in_axes maps 1-to-1 with the arguments of _solve_trajopt_core
#         in_axes=(
#             None,  # robot
#             None,  # robot_coll
#             None,  # world_coll
#             None,  # target_link_index
#             0,     # start_position (Batched)
#             0,     # start_wxyz (Batched)
#             0,     # end_position (Batched)
#             0,     # end_wxyz (Batched)
#             None,  # timesteps (Shared)
#             None,  # dt (Shared)
#         ),
#     )(
#         robot,
#         robot_coll,
#         world_coll,
#         target_link_index,
#         start_positions,
#         start_wxyzs,
#         end_positions,
#         end_wxyzs,
#         timesteps,
#         dt,
#     )

# 1. REMOVE THE DECORATOR HERE
def solve_trajopt_batched(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_link_index: int,
    start_positions: jax.Array,
    start_wxyzs: jax.Array,
    end_positions: jax.Array,
    end_wxyzs: jax.Array,
    timesteps: int,
    dt: float,
) -> jax.Array:
    """
    Batched solver.
    """
    return jax.vmap(
        _solve_trajopt_core,
        in_axes=(
            None,  # robot
            None,  # robot_coll
            None,  # world_coll
            None,  # target_link_index
            0,     # start_position (Batched)
            0,     # start_wxyz (Batched)
            0,     # end_position (Batched)
            0,     # end_wxyz (Batched)
            None,  # timesteps (Shared)
            None,  # dt (Shared)
        ),
    )(
        robot,
        robot_coll,
        world_coll,
        target_link_index,
        start_positions,
        start_wxyzs,
        end_positions,
        end_wxyzs,
        timesteps,
        dt,
    )

# 2. ADD EXPLICIT JIT CALL HERE
# This avoids the "missing 'fun'" TypeError by passing the function explicitly.
solve_trajopt_batched = jax.jit(
    solve_trajopt_batched, 
    static_argnames=["timesteps", "dt"]
)

def solve_trajopt(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_link_name: str,
    start_position: ArrayLike,
    start_wxyz: ArrayLike,
    end_position: ArrayLike,
    end_wxyz: ArrayLike,
    timesteps: int,
    dt: float,
) -> ArrayLike:
    """Wrapper for single problem that handles numpy inputs and string lookup."""
    target_link_index = robot.links.names.index(target_link_name)
    
    # Ensure inputs are JAX arrays
    start_position = jnp.array(start_position)
    start_wxyz = jnp.array(start_wxyz)
    end_position = jnp.array(end_position)
    end_wxyz = jnp.array(end_wxyz)

    sol = _solve_trajopt_core(
        robot, robot_coll, world_coll, target_link_index,
        start_position, start_wxyz, end_position, end_wxyz,
        timesteps, dt
    )
    return onp.array(sol) # Return numpy for consistency with original API


@jdc.jit
def solve_iks_with_collision(
    # ... (Keep this function exactly as it was in your snippet) ...
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,
    world_coll_list: Sequence[pk.collision.CollGeom],
    target_link_index: int,
    target_position_0: jax.Array,
    target_wxyz_0: jax.Array,
    target_position_1: jax.Array,
    target_wxyz_1: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    # ... (Implementation from your snippet) ...
    # (I'm assuming you have the content from your previous message here)
    # The only change needed here is to ensure it's available for _solve_trajopt_core
    
    # ... [PASTE YOUR EXISTING solve_iks_with_collision HERE] ...
    
    joint_var_0 = robot.joint_var_cls(0)
    joint_var_1 = robot.joint_var_cls(1)
    joint_vars = robot.joint_var_cls(jnp.arange(2))
    vars = [joint_vars]

    factors = [
        pk.costs.pose_cost(
            robot,
            joint_var_0,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz_0), target_position_0
            ),
            jnp.array(target_link_index),
            jnp.array([5.0] * 3),
            jnp.array([1.0] * 3),
        ),
        pk.costs.pose_cost(
            robot,
            joint_var_1,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz_1), target_position_1
            ),
            jnp.array(target_link_index),
            jnp.array([5.0] * 3),
            jnp.array([1.0] * 3),
        ),
    ]
    
    factors.extend([
        pk.costs.limit_cost(
            jax.tree.map(lambda x: x[None], robot),
            joint_vars,
            jnp.array(100.0),
        ),
        pk.costs.rest_cost(
            joint_vars,
            jnp.array(joint_vars.default_factory()[None]),
            jnp.array(0.001),
        ),
        pk.costs.self_collision_cost(
            jax.tree.map(lambda x: x[None], robot),
            jax.tree.map(lambda x: x[None], coll),
            joint_vars,
            0.02,
            5.0,
        ),
    ])
    
    factors.extend([
        pk.costs.world_collision_cost(
            jax.tree.map(lambda x: x[None], robot),
            jax.tree.map(lambda x: x[None], coll),
            joint_vars,
            jax.tree.map(lambda x: x[None], world_coll),
            0.05,
            10.0,
        ) for world_coll in world_coll_list
    ])

    @jaxls.Cost.create_factory(name="JointSimilarityCost")
    def joint_similarity_cost(vals, var_0, var_1):
        return ((vals[var_0] - vals[var_1]) * 0.01).flatten()

    factors.append(joint_similarity_cost(joint_var_0, joint_var_1))

    sol = jaxls.LeastSquaresProblem(factors, vars).analyze().solve(verbose=False)
    return sol[joint_var_0], sol[joint_var_1]


from typing import Dict, Tuple, Optional

def solve_waypoint_trajopt(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_link_name: str,
    start_cfg: ArrayLike,              # Fixed joint config for t=0
    waypoints: Dict[int, Tuple[ArrayLike, ArrayLike]], # {t: (pos, wxyz)}
    timesteps: int,
    dt: float,
) -> onp.ndarray:
    
    # 1. Setup
    target_link_index = robot.links.names.index(target_link_name)
    start_cfg = jnp.array(start_cfg)
    
    # Initialize trajectory with the start configuration repeated
    # (Since we don't have an end_cfg to interpolate towards, this is a safe default)
    init_traj = jnp.tile(start_cfg, (timesteps, 1))
    
    traj_vars = robot.joint_var_cls(jnp.arange(timesteps))
    
    # Batch wrappers
    robot_b = jax.tree.map(lambda x: x[None], robot)
    robot_coll_b = jax.tree.map(lambda x: x[None], robot_coll)

    factors: list[jaxls.Cost] = []

    # --- 2. START CONFIGURATION CONSTRAINT (t=0) ---
    # Force the trajectory to begin exactly at start_cfg
    factors.append(
        jaxls.Cost(
            lambda vals, var: ((vals[var] - start_cfg) * 100.0).flatten(),
            (robot.joint_var_cls(0),),
            name="start_cfg_constraint",
        )
    )

    # --- 3. WAYPOINT CONSTRAINTS (Intermediate + End) ---
    # Loop through the dictionary and add a pose cost for each requested timestep
    for t, (pos, wxyz) in waypoints.items():
        if t < 0 or t >= timesteps:
            continue # specific check to avoid index errors
            
        factors.append(
            pk.costs.pose_cost(
                robot,
                robot.joint_var_cls(t),
                jaxlie.SE3.from_rotation_and_translation(
                    jaxlie.SO3(jnp.array(wxyz)), jnp.array(pos)
                ),
                jnp.array(target_link_index),
                jnp.array([20.0] * 3), # High weight for Position
                jnp.array([10.0] * 3), # Medium weight for Rotation
            )
        )

    # --- 4. STANDARD COSTS (Collision, Smoothness) ---
    
    # Collision Avoidance
    def compute_world_coll_residual(vals, r, rc, wc, prev, curr):
        coll = rc.get_swept_capsules(r, vals[prev], vals[curr])
        dist = pk.collision.collide(coll.reshape((-1, 1)), wc.reshape((1, -1)))
        colldist = pk.collision.colldist_from_sdf(dist, 0.1)
        return (colldist * 20.0).flatten()

    for world_coll_obj in world_coll:
        factors.append(
            jaxls.Cost(
                compute_world_coll_residual,
                (
                    robot_b, robot_coll_b, jax.tree.map(lambda x: x[None], world_coll_obj),
                    robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                    robot.joint_var_cls(jnp.arange(1, timesteps)),
                ),
                name="World Collision",
            )
        )

    # Regularization
    factors.extend([
        # Smoothness (minimize velocity)
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            jnp.array([0.1])[None],
        ),
        # Joint Limits
        pk.costs.limit_cost(
            robot_b, traj_vars, jnp.array([100.0])[None],
        ),
        # Stay close to default pose (regularization)
        pk.costs.rest_cost(
            traj_vars, traj_vars.default_factory()[None], jnp.array([0.01])[None],
        ),
    ])

    # 5. Solve
    solution = (
        jaxls.LeastSquaresProblem(factors, [traj_vars])
        .analyze()
        .solve(initial_vals=jaxls.VarValues.make((traj_vars.with_value(init_traj),)))
    )
    return onp.array(solution[traj_vars])