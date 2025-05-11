# motion-planning

## High-Level Architecture
- **Perception**:
    - I used color thresholding to get the 2D position of objects in the image
    - I then used camera intrinsics and extrinsics to map these 2D image coordinates to 3D real-world coordinates.
- **Control**:
    - I developed a Proportional-Integral (PI) controller. I found that derivative control wasn't essential as the system wasn't prone to overly drastic movements.
    - This controller was necessary because the simulation accepts `dx, dy, dz, rpy` (i.e., changes in position and roll, pitch, yaw) as input, rather than absolute `x, y, z, rpy` coordinates.
    - To simplify interactions, I designed this controller so I can provide a target pose, and it will autonomously generate the control actions to reach it.
- **Waypoint Generation**:
    - I generated intermediate waypoints to create paths to/from objects
    - These intermediate waypoints were assembled from motion primitives to make a single pick and a single place sequence that can be repeated ad infinitum.

## Simulation Flow

1.  `get_stacking_order_from_user` (from `motion_planning.utils`) prompts the user for the desired order of blocks to stack (user has to input them to the command line)
2. The robot's initial pose is reset using `task_executor.reset_pose()`.
3.  A `CameraProcessor` (from `motion_planning.camera_processor`) is initialized. It uses the simulator's camera feed. The position of the base block (the first block in the `stack_order`) is determined using this processor.
4.  **Stacking Loop**:
    *   The script iterates through the remaining blocks in the `stack_order`.
    *   For each block:
        *   `camera_processor.get_block_position_from_color()` is called to find the current 3D position of the block to be picked.
        *   The `target_pos` for placing the block is calculated by adding the `OBJECT_HEIGHT` cumulatively to the `base_pos`.
        *   `task_executor.perform_pick_and_place()` is called with the `pick_location` and `target_location` to execute the stacking maneuver. This uses the `TaskExecutor` class, which is a wrapper that internally uses the `TrajectoryController` and `WaypointPlanner` (from `motion_planning.waypoint_planner`) to simplify the generation and following of a path or high level task.

## What I would do given some more time:
- Focus on implementing smooth splines for more fluid intermediate waypoint generation. While the current simplified version is reliable, smoother paths could enhance speed and elegance.
- Address yaw perception. Accurately inferring yaw from box depth or color data proved more challenging than anticipated, and I couldn't complete this aspect within the timeframe.
- Ideally, I'd integrate continuous camera feedback to dynamically adjust waypoints or the overall path, enabling the system to adapt to unforeseen obstacles. Although not strictly necessary for this specific task, it would be invaluable for detecting success/failure and triggering retry mechanisms, for instance.
