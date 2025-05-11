import numpy as np

from motion_planning.camera_processor import CameraProcessor
from motion_planning.simulator import Simulator
from motion_planning.task_executor import TaskExecutor
from motion_planning.trajectory_controller import TrajectoryController
from motion_planning.utils import get_stacking_order_from_user

OBJECT_HEIGHT = 0.05  # Height of the blocks (meters)

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

    # Now, we are ready to start stacking
    # First we have to get the base position
    # We use the camera processor to get the positions of each block
    # but we only get the position when we actually need it (in case something
    # gets moved in the meantime)
    camera_processor = CameraProcessor(simulator=sim)
    base_pos = camera_processor.get_block_position_from_color(obs, stack_order[0])
    print(f"Base position: {base_pos}")

    for i, color in enumerate(stack_order[1:]):
        # Get the position of the block we want to pick
        pick_pos = camera_processor.get_block_position_from_color(obs, color)
        target_pos = base_pos.copy()
        target_pos[2] += OBJECT_HEIGHT * (i + 1)  # Stack height
        print(f"Moving {color.upper()} to {target_pos} from {pick_pos}")
