import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

urdf_path = "leader.urdf"
mesh_dir = "meshes"

robot = RobotWrapper.BuildFromURDF(urdf_path, [mesh_dir])
model = robot.model
data = model.createData()

# Print all frames
for i, frame in enumerate(model.frames):
    print(f"{i}: {frame.name} (type={frame.type})")

# Neutral configuration
q = pin.neutral(model)

# FK
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)

# Example: end-effector (replace with your frame name!)
ee_name = " trigger_1"
if ee_name in [f.name for f in model.frames]:
    ee_id = model.getFrameId(ee_name)
    ee_pose = data.oMf[ee_id]
    print(f"\nEnd-effector ({ee_name}) pose:")
    print("  Position:", ee_pose.translation)
    print("  Rotation:\n", ee_pose.rotation)
else:
    print(f"\nNo frame named '{ee_name}' in this URDF.")
