# Disable JAX prealloc etc
import os
from pathlib import Path

import viser
from viser.extras import ViserUrdf

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Standard Library
import argparse
import time

# Third Party
import numpy as np
import torch

# set seeds
torch.manual_seed(2)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


import xml.etree.ElementTree as ET
from io import StringIO

import jax
import jaxlie
import jaxls
import pyroki as pk
import yourdfpy
from jax import lax
from jax import numpy as jnp
from robot_descriptions.loaders.yourdfpy import load_robot_description


def newton_raphson(f, x, iters):
    """Use the Newton-Raphson method to find a root of the given function."""

    def update(x, _):
        y = x - f(x) / jax.grad(f)(x)
        return y, None

    x, _ = lax.scan(update, 1.0, length=iters)
    return x


def roberts_sequence(num_points, dim, root):
    # From https://gist.github.com/carlosgmartin/1fd4e60bed526ec8ae076137ded6ebab.
    basis = 1 - (1 / root ** (1 + jnp.arange(dim)))

    n = jnp.arange(num_points)
    x = n[:, None] * basis[None, :]
    x, _ = jnp.modf(x)

    return x


class PyrokiIkBeamHelper:
    def __init__(self, visualize: bool = False):
        self.visualize = visualize
        # Get the Panda robot. We fix the prismatic (gripper) joints. This is to
        # match cuRobo, it makes a very small runtime difference.
        # urdf = load_robot_description("panda_description")
        # xml_tree = urdf.write_xml()
        # for joint in xml_tree.findall('.//joint[@type="prismatic"]'):
        #     joint.set("type", "fixed")
        #     for tag in ("axis", "limit", "dynamics"):
        #         child = joint.find(tag)
        #         if child is not None:
        #             joint.remove(child)
        # xml_str = ET.tostring(xml_tree.getroot(), encoding="unicode")
        # buf = StringIO(xml_str)
        # urdf = yourdfpy.URDF.load(buf)
        # assert urdf.validate()

        import yourdfpy

        # yourdfpy => pyrokiurdf = yourdfpy.
        urdf = yourdfpy.URDF.load(
            Path("/home/nvidia/repos/boros/boros/piper_arm/assets/piper.urdf"),
            mesh_dir=Path("/home/nvidia/repos/boros/boros/piper_arm/assets/meshes"),
        )
        ee_link_name = "eef"

        # Create robot.
        robot = pk.Robot.from_urdf(urdf)

        target_link_index = jnp.array(robot.links.names.index(ee_link_name))

        self.robot = robot
        exp = robot.joints.num_actuated_joints
        self.root = newton_raphson(lambda x: x ** (exp + 1) - x - 1, 1.0, 10_000)
        self.target_link_index = target_link_index

    def solve_ik(self, target_wxyz: jax.Array, target_position: jax.Array) -> jax.Array:
        num_seeds_init: int = 64
        num_seeds_final: int = 4

        total_steps: int = 16
        init_steps: int = 6

        def solve_one(
            initial_q: jax.Array, lambda_initial: float | jax.Array, max_iters: int
        ) -> tuple[jax.Array, jaxls.SolveSummary]:
            """Solve IK problem with a single initial condition. We'll vmap
            over initial_q to solve problems in parallel."""
            joint_var = robot.joint_var_cls(0)
            factors = [
                # pk.costs.pose_cost(
                pk.costs.pose_cost_analytic_jac(
                    robot,
                    joint_var,
                    jaxlie.SE3.from_rotation_and_translation(
                        jaxlie.SO3(target_wxyz), target_position
                    ),
                    self.target_link_index,
                    pos_weight=50.0,
                    ori_weight=10.0,
                ),
                pk.costs.limit_cost(
                    robot,
                    joint_var,
                    weight=100.0,
                ),
                pk.costs.manipulability_cost(
                    robot,
                    joint_var,
                    self.target_link_index,
                    0.0,
                ),
            ]
            sol, summary = (
                jaxls.LeastSquaresProblem(factors, [joint_var])
                .analyze()
                .solve(
                    initial_vals=jaxls.VarValues.make(
                        [joint_var.with_value(initial_q)]
                    ),
                    verbose=False,
                    linear_solver="dense_cholesky",
                    termination=jaxls.TerminationConfig(
                        max_iterations=max_iters,
                        early_termination=False,
                    ),
                    trust_region=jaxls.TrustRegionConfig(lambda_initial=lambda_initial),
                    return_summary=True,
                )
            )
            return sol[joint_var], summary

        vmapped_solve = jax.vmap(solve_one, in_axes=(0, 0, None))

        # Create initial seeds, but this time with quasi-random sequence.
        robot = self.robot
        initial_qs = robot.joints.lower_limits + roberts_sequence(
            num_seeds_init, robot.joints.num_actuated_joints, self.root
        ) * (robot.joints.upper_limits - robot.joints.lower_limits)

        # Optimize the initial seeds.
        initial_sols, summary = vmapped_solve(
            initial_qs, jnp.full(initial_qs.shape[:1], 10.0), init_steps
        )

        # Get the best initial solutions.
        best_initial_sols = jnp.argsort(
            summary.cost_history[jnp.arange(num_seeds_init), -1]
        )[:num_seeds_final]

        # Optimize more for the best initial solutions.
        best_sols, summary = vmapped_solve(
            initial_sols[best_initial_sols],
            summary.lambda_history[jnp.arange(num_seeds_init), -1][best_initial_sols],
            total_steps - init_steps,
        )
        return best_sols[
            jnp.argmin(
                summary.cost_history[jnp.arange(num_seeds_final), summary.iterations]
            )
        ]

    def forward_kinematics(self, q: jax.Array | np.ndarray) -> jax.Array:
        return self.robot.forward_kinematics(jnp.asarray(q))[self.target_link_index]


# Batched helpers for IK and FK.
ik_beam = PyrokiIkBeamHelper(visualize=True)
# batched_ik = jax.jit(jax.vmap(ik_beam.solve_ik))
batched_ik = jax.vmap(ik_beam.solve_ik)
# batched_fk = jax.jit(jax.vmap(ik_beam.forward_kinematics))
batched_fk = jax.vmap(ik_beam.forward_kinematics)


def _get_poses_and_quats(mats: np.ndarray):
    from scipy.spatial.transform import Rotation as R

    # positions: (N,3)
    positions = mats[:, :3, 3]

    # rotations → quaternions (w,x,y,z)
    rot_mats = mats[:, :3, :3]
    r = R.from_matrix(rot_mats)
    # scipy gives (x,y,z,w)
    xyzw = r.as_quat()
    quaternions = np.stack(
        [
            xyzw[:, 3],  # w
            xyzw[:, 0],  # x
            xyzw[:, 1],  # y
            xyzw[:, 2],  # z
        ],
        axis=1,
    )

    return positions, quaternions


if __name__ == "__main__":
    # Test the IK solver.
    np.set_printoptions(precision=3, suppress=True)
    pose = np.array(
        [
            [1, 0, 0.0, -0.1],
            [0, 1.0, 0.0, 0.0],
            [-0.0, 0.0, 1, 0.2],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    pose = np.expand_dims(pose, axis=0)  # Add batch dimension
    positions, quaternions = _get_poses_and_quats(pose)
    positions = np.squeeze(positions, axis=0)  # Remove batch dimension
    quaternions = np.squeeze(quaternions, axis=0)  # Remove batch dimension

    print("Target position:", positions)
    print("Target quaternion:", quaternions)
    target_position = jnp.array(positions)
    target_wxyz = jnp.array(quaternions)

    solution = ik_beam.solve_ik(
        target_position=target_position, target_wxyz=target_wxyz
    )
    print("IK solution:", solution)