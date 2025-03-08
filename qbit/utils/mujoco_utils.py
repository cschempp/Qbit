from mujoco import mj_id2name


def print_object_names(model, obj_type, count, label):
    """
    Prints the names for objects of a given type from the model.
    """
    print(f"{label}:")
    for i in range(count):
        # Retrieve the name of the object using its type and id.
        name = mj_id2name(model, obj_type, i)
        if name is not None:
            print(f"  - {name}")
        else:
            print(f"  - [unnamed] (id {i})")


def convert_quat_to_wxyz(quat):
    """
    Convert the quaternion format between wxyz for mujoco and xyzw.
    """
    return [quat[3], quat[0], quat[1], quat[2]]
    
def convert_quat_to_xyzw(quat):
    """
    Convert the quaternion format between wxyz for mujoco and xyzw.
    """
    return [quat[1], quat[2], quat[3], quat[0]]

