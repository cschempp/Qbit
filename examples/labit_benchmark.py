"""
Position based insertion in development mode and use for following tests:
 - Collect the training data for insertion-net
 - Randomize the pyhsical parameter in MuJoCo to compare the force distribution with the real robot
 - Compare the mesh decomposition with different mesh scale
 - Test the surface toughness with the sphere-based method
"""
from copy import deepcopy

import numpy as np
import mujoco
import mujoco.viewer

import time

from qbit.controllers.eef_position_controller import EEFPositionController
from qbit.utils.tf_utils import T
from qbit.utils.mj_viewer_utils import update_view_camera_parameter
from qbit.utils.mujoco_utils import get_relative_pose, get_body_pose_in_world, print_object_names
from qbit.sim_envs.mujoco_env_insertion import MjEnvInsertion, MujocoEnvBase


NUM_RUNS = 300

RESULT_DIR = "/workspace/examples/experiment_results/position_based/exp_labit_benchmark"

SIM_TIMESTEP = 0.0005

NN_CONTROL_DT = 0.01
ADMITTANCE_CONTROL_T = 0.01
JOINT_POSITION_CONTROL_T = 0.001  # Second

POS_RANDOM_LIMIT = 0.001 # meter
ROT_RANDOM_LIMIT = 1.5 # degree


class PositionBasedInsertion(MujocoEnvBase):
    
    
    def __init__(self,
                 task_env_config_path: str,
                 sim_timestep: float = 0.001,
                 rendering_timestep: float = 0.033,
                 rt_factor: float = 0.0,
                 headless: bool = True,
                 server_modus: bool = False,
                 ):
        
        super(PositionBasedInsertion, self).__init__(
            task_env_config_path,
            sim_timestep,
            rendering_timestep,
            rt_factor,
            headless,
            server_modus,
        )
        
        #self.data_eva = DataRecording(task_env_config_path=task_env_config_path)

    
    def termination(self, 
                    q_current: T,
                    q_goal: T,
                    threshold: float = 0.001
                    ) -> bool:


        error = np.abs(q_current - q_goal)
        with np.printoptions(precision=4, floatmode="fixed", suppress=True):
            print("current error: {}".format(error))

        if  (all(error <= threshold)):
            return True

        return False

    def mocap_move_pose(self, viewer, _goal_pose_T: T = None,):
        integration_dt: float = 1.0

        # Damping term for the pseudoinverse. This is used to prevent joint velocities from
        # becoming too large when the Jacobian is close to singular.
        damping: float = 1e-4

        # Simulation timestep in seconds.
        # dt: float = 0.002

        # Maximum allowable joint velocity in rad/s. Set to 0 to disable.
        max_angvel = np.pi

        site_id = self._mj_model.site("attachment_site").id

        # Get the dof and actuator ids for the joints we wish to control.
        joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        actuator_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow",
            "wrist_1",
            "wrist_2",
            "wrist_3",
        ]
        dof_ids = np.array([self._mj_model.joint(name).id for name in joint_names])
        # Note that actuator names are the same as joint names in this case.
        actuator_ids = np.array([self._mj_model.actuator(name).id for name in actuator_names])

        # Initial joint configuration saved as a keyframe in the XML file.
        # key_id = self._mj_model.key("home").id

        # Mocap body we will control with our mouse.
        mocap_id = self._mj_model.body("target").mocapid[0]

        # Pre-allocate numpy arrays.
        jac = np.zeros((6, self._mj_model.nv))
        diag = damping * np.eye(6)
        error = np.zeros(6)
        error_pos = error[:3]
        error_ori = error[3:]
        site_quat = np.zeros(4)
        site_quat_conj = np.zeros(4)
        error_quat = np.zeros(4)        

        while True:
            # Set the target position of the end-effector site.
            # self._mj_data.mocap_pos[mocap_id, 0:2] = circle(self._mj_data.time, 0.1, 0.5, 0.0, 0.5)

            self._mj_data.mocap_pos[mocap_id, :] = _goal_pose_T.translation
            self._mj_data.mocap_quat[mocap_id, :] = _goal_pose_T.quaternion

            # Position error.
            error_pos[:] = self._mj_data.mocap_pos[mocap_id] - self._mj_data.site(site_id).xpos

            # Orientation error.
            mujoco.mju_mat2Quat(site_quat, self._mj_data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, self._mj_data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)

            # Get the Jacobian with respect to the end-effector site.
            mujoco.mj_jacSite(self._mj_model, self._mj_data, jac[:3], jac[3:], site_id)

            # Solve system of equations: J @ dq = error.
            dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)

            # Scale down joint velocities if they exceed maximum.
            if max_angvel > 0:
                dq_abs_max = np.abs(dq).max()
                if dq_abs_max > max_angvel:
                    dq *= max_angvel / dq_abs_max

            # Integrate joint velocities to obtain joint positions.
            q = self._mj_data.qpos.copy()
            mujoco.mj_integratePos(self._mj_model, q, dq, integration_dt)
            
            min, max = self._mj_model.jnt_range.T
            jointrangelen = len(q)
            min = np.append(np.array(min), np.zeros(jointrangelen - len(min)))
            max = np.append(np.array(max), np.zeros(jointrangelen - len(max)))
            
            # Set the control signal.
            np.clip(q, min, max, out=q)
            self._mj_data.ctrl[actuator_ids] = q[dof_ids]

            # Step the simulation.
            mujoco.mj_step(self._mj_model, self._mj_data)

            if viewer is not None:
                viewer.sync()
            
            with np.printoptions(precision=4, floatmode="fixed", suppress=True):
                print("[ROBOT] error position: {:.4f}, error orientation: {:.4f}".format(
                    np.linalg.norm(error_pos), np.linalg.norm(error_ori)))

            if np.linalg.norm(error_pos) < 0.0003 and np.linalg.norm(error_ori) < 0.001:
                print("[ROBOT] reached goal pose")
                break

    def move_pose(self,
                  viewer,
                  _goal_pose_T: T = None,
                  qgoal: np.ndarray = None
                  ):
        
        q_init = self.robot.get_current_joint_state()[0]
        if qgoal is not None:
            q_goal = qgoal
        else:
            q_goal = self.robot._eef_position_controller.ik.ik(_goal_pose_T.matrix, q_init) 
        
        t_max = 5.0  # seconds
        i = 0
        # self._mj_data.ctrl[0:6] = q_goal
        while 1:
            # get states
            q_current, _ = self.robot.get_current_joint_state()
            current_eef_pose_T = self.robot.get_eef_pose_in_base_frame()
            # current_joint_state = self.robot.get_current_joint_state()
            # measured_wrench = self.robot.get_fts_data(transform_to_base=True)
            
            # self._mj_data.ctrl[0:6] = q_goal
            # self._mj_data.qpos[6] = self._gripper_qpos
            # self._mj_data.qpos[10] = self._gripper_qpos

            # t = i*self._sim_timestep/t_max
            # t = np.minimum(t, 1.0)
            # target = q_init * (1-t) + q_goal * t

            # Check the termination condition
            if self.termination(q_current, q_goal):
                # self.data_eva.save()
                print("reached goal pose {}".format(q_current))
                break
            
            # EEF Position control
            next_eef_goal = self.robot._eef_position_controller.eef_position_control(
                current_eef_pose = current_eef_pose_T,
                target_eef_pose = _goal_pose_T,
                q_init = self.robot.get_current_joint_state()[0],
                return_q_cmd=False
            )

            # Joint position control
            self.robot.move_to_eef_pose(
                viewer=viewer,
                eef_pose=next_eef_goal.get_pos_quat_list(quat_format='xyzw'),
                # eef_pose=goal_pose_T.get_pos_quat_list(quat_format='xyzw'),
                # qpos_thresh=1.0 * np.pi/180,
                executing=False
            )

            
            for _ in range(1):
                self.robot.spin()
                self.step_mj_simulation()

            if viewer != None:    
                viewer.sync()
            i += 1

        # self._mj_data.ctrl[0:6] = q_goal
        # self._mj_data.ctrl[0:6] = self.robot.get_current_joint_state()[0]
        pos_a, quat_a =  get_body_pose_in_world(self._mj_model, self._mj_data, "tool0")
        print("tool0 in world frame: pos {}, quat {}".format(pos_a, quat_a))  


    def set_gripper_position(self, position: float, viewer):
        """
        Set the gripper position (width between fingers).
        position: float, range from [0.0, 0.085] meter [fully closed, fully open]
        """

        while True:
            
            self._mj_data.ctrl[6] = 255 - 255 * position / 0.085
            mujoco.mj_step(self._mj_model, self._mj_data)

            with np.printoptions(precision=4, floatmode="fixed", suppress=True):
                print("[GRIPPER] qvel: {}".format(self._mj_data.qvel[6:13]))
            
            if viewer is not None:
                viewer.sync()
            
            if any(abs(self._mj_data.qvel[6:13]) < 0.003):
                break
        
        print("[GRIPPER] gripper position set. Extra simulation steps for safety.")
        for _ in range(int(.1//self._sim_timestep)):
            mujoco.mj_step(self._mj_model, self._mj_data)
    

    def apply_gravity_compensation(self):
        # Name of bodies we wish to apply gravity compensation to.
        body_names = [
            "base",
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
            "tool0"
        ]
        body_ids = [self._mj_model.body(name).id for name in body_names]
        
        self._mj_model.body_gravcomp[body_ids] = 1.0
    

    def get_body_grasp_pose(self, body_name: str, z_offset: float):
        body_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY.value, body_name)

        body_pos = self._mj_data.xpos[body_id]                  # world position of body
        body_R   = self._mj_data.xmat[body_id].reshape(3,3)     # 3×3 world-from-body rotation
        body_quat = self._mj_data.xquat[body_id, :] 

        z_body_in_world = body_R[:, 2]              # local z-axis expressed in world frame

        # Ensure the returned orientation has the local z-axis pointing downwards
        if z_body_in_world[2] > 0:
            # Rotate 180° around the body x-axis (in body frame) to flip local z.
            # This is achieved by post-multiplying the body rotation by diag([1, -1, -1]).
            R_flip = np.diag([1.0, -1.0, -1.0])
            R_new = body_R @ R_flip
            # Convert the new rotation matrix to a quaternion using MuJoCo utility.
            body_quat_new = np.zeros(4, dtype=float)
            mujoco.mju_mat2Quat(body_quat_new, R_new.reshape(9))
            body_quat = body_quat_new
            z_body_used = R_new[:, 2]
        else:
            z_body_used = z_body_in_world

        p_shifted_world = body_pos + z_offset * z_body_used * np.sign(z_body_used[2])

        return p_shifted_world, body_quat


    def labit_policy(self, viewer = None):
        
        body_name = "plug_inside_loose_2_body" #"positioning_pin_d5_20_1_body"
        # close gripper# close gripper
        self.set_gripper_position(0.02, viewer)  
        
        # move to body
        pos, quat = self.get_body_grasp_pose(body_name=body_name, z_offset=0.1)
        goal_pose_T = T(translation=pos,
                          quaternion=quat)
        self.mocap_move_pose(viewer=viewer, _goal_pose_T=goal_pose_T)

        # move to grasp pose (body coordinate frame in between fingertips)
        pos, quat = self.get_body_grasp_pose(body_name=body_name, z_offset=0.0092)
        goal_pose_T_2 = T(translation=pos,
                          quaternion=quat)
        self.mocap_move_pose(_goal_pose_T=goal_pose_T_2, viewer=viewer)

        # close the gripper to grasp
        self.set_gripper_position(0.011, viewer)

        # move away along object z axis
        pos, quat = self.get_body_grasp_pose(body_name=body_name, z_offset=0.1)  
        goal_pose_T_3 = T(translation=pos,
                          quaternion=quat)
        self.mocap_move_pose(_goal_pose_T=goal_pose_T_3, viewer=viewer)

        while 1:
            self.step_mj_simulation()

            if viewer != None:
                viewer.sync()
        return

    def exec_labit(self):
        """
        Main function to execute the LABIT benchmark task.
        """

        # self.apply_gravity_compensation() # wont work; did it in xml
        with mujoco.viewer.launch_passive(self._mj_model, self._mj_data, show_left_ui=False, show_right_ui=False) as viewer:
            
            self.update_view_scale()
            self.update_view_opt(viewer)
            update_view_camera_parameter(viewer, view_type="labit_benchmark")

            print_object_names(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, self._mj_model.nbody, "Bodies in the model")

            self.labit_policy(viewer=viewer)


    def exec_labit_headless(self):
        self.labit_policy()

        return


  

if __name__ == "__main__":
    
    task_env_config_path = "/workspace/qbit/configs/envs/ur5e_labit_benchmark.yaml"
    
    mj = PositionBasedInsertion(
        task_env_config_path=task_env_config_path,
        server_modus=True,
        sim_timestep=SIM_TIMESTEP,
        )
    mj.exec_labit()
    # mj.exec_labit_headless()