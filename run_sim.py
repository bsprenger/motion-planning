import numpy as np

from motion_planning.simulator import Simulator
from motion_planning.task_executor import TaskExecutor
from motion_planning.trajectory_controller import TrajectoryController
from motion_planning.utils import get_stacking_order_from_user

if __name__ == "__main__":
    sim = Simulator()
    obs = sim.reset()
    sim.render()

    # We start by getting a stacking order from the user
    stack_order = get_stacking_order_from_user()

    # We will need a controller to actually move the robot towards a target pose
    # The controller is PI-based (for simplicity, D is not used -- performance
    # is good enough without it)
    # The controller has all sorts of settings for gains/clip limits etc.,
    # but the defaults are good for this case
    ctrl = TrajectoryController()

    # Next, we create a TaskExecutor, which is a wrapper that simplifies the
    # interface between the simulator and the trajectory controller
    # Essentially, it just takes a target pose and keeps applying control inputs
    # until the target pose is reached
    task_executor = TaskExecutor(simulator=sim, controller=ctrl)
    obs, _ = task_executor.reset_pose(obs, max_steps=100)

    for i in range(1000):
        # Set the desired 6DOF position of the end effector + gripper position
        action = np.random.randn(*sim.action_spec[0].shape) * 0.1
        observation = sim.step(action)

        if i == 0:
            print("-- Observation --")
            for k, v in observation.items():
                print(f"{k}: Shape [{v.shape}], Type [{v.dtype}]")
            print("-- Action --")
            action_min, action_max = sim.action_spec
            print(
                f"Action: Shape [{action.shape}], Type [{action.dtype}], Min [{action_min}], Max [{action_max}]"
            )
        sim.render()
