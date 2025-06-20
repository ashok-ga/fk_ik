import numpy as np
import jax
from jax import lax
from jax import numpy as jnp

import pyroki as pk
import yourdfpy

def newton_raphson(f, x, iters):
    """Use the Newton-Raphson method to find a root of the given function."""
    def update(x, _):
        y = x - f(x) / jax.grad(f)(x)
        return y, None
    x, _ = lax.scan(update, 1.0, length=iters)
    return x

def roberts_sequence(num_points, dim, root):
    basis = 1 - (1 / root ** (1 + jnp.arange(dim)))
    n = jnp.arange(num_points)
    x = n[:, None] * basis[None, :]
    x, _ = jnp.modf(x)
    return x

class PyrokiIkBeamHelper:
    def __init__(self, urdf_path, mesh_dir, ee_link_name):
        self.urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=mesh_dir)
        self.robot = pk.Robot.from_urdf(self.urdf)
        self.ee_link_name = ee_link_name
        self.target_link_index = jnp.array(self.robot.links.names.index(ee_link_name))
        exp = self.robot.joints.num_actuated_joints
        self.root = newton_raphson(lambda x: x ** (exp + 1) - x - 1, 1.0, 10_000)

    def solve_ik(self, target_wxyz, target_position):
        num_seeds_init, num_seeds_final = 64, 4
        total_steps, init_steps = 16, 6
        robot = self.robot
        target_link_index = self.target_link_index

        def solve_one(initial_q, lambda_initial, max_iters):
            import jaxlie, jaxls, pyroki as pk
            joint_var = robot.joint_var_cls(0)
            factors = [
                pk.costs.pose_cost_analytic_jac(
                    robot,
                    joint_var,
                    jaxlie.SE3.from_rotation_and_translation(
                        jaxlie.SO3(target_wxyz), target_position
                    ),
                    target_link_index,
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
                    target_link_index,
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
        initial_qs = robot.joints.lower_limits + roberts_sequence(
            num_seeds_init, robot.joints.num_actuated_joints, self.root
        ) * (robot.joints.upper_limits - robot.joints.lower_limits)

        initial_sols, summary = vmapped_solve(
            initial_qs, jnp.full(initial_qs.shape[:1], 10.0), init_steps
        )
        best_initial_sols = jnp.argsort(
            summary.cost_history[jnp.arange(num_seeds_init), -1]
        )[:num_seeds_final]
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

    def forward_kinematics(self, q):
        return self.robot.forward_kinematics(np.asarray(q))[self.target_link_index]
