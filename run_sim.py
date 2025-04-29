import numpy as np
from motion_planning.simulator import Simulator

if __name__ == "__main__":
    sim = Simulator()

    sim.reset()

    # 6DOF position of the camera
    camera_translation, camera_rotation = sim.get_camera_transform()
    print("Camera Translation:", camera_translation)
    print("Camera Rotation:", camera_rotation)

    # Path to model of the robot arm
    print("Robot MJCF Path:", sim.get_robot_mjcf_path())

    for i in range(1000):
        action = np.random.randn(*sim.action_spec[0].shape) * 0.1
        observation = sim.step(action)  

        if i == 0:
            print("-- Observation --")
            for k, v in observation.items():
                print(f"{k}: Shape [{v.shape}], Type [{v.dtype}]")
            print("-- Action --")
            action_min, action_max = sim.action_spec
            print(f"Action: Shape [{action.shape}], Type [{action.dtype}], Min [{action_min}], Max [{action_max}]")
        sim.render() 