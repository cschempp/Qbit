"""
Position based insertion in development mode and use for following tests:
 - Collect the training data for insertion-net
 - Randomize the pyhsical parameter in MuJoCo to compare the force distribution with the real robot
 - Compare the mesh decomposition with different mesh scale
 - Test the surface toughness with the sphere-based method
"""

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

SIM_TIMESTEP = 0.0001

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
                    threshold: float = 0.01
                    ) -> bool:


        error = np.abs(q_current - q_goal)
        with np.printoptions(precision=4, floatmode="fixed", suppress=True):
            print("current error: {}".format(error))

        if  all(error < threshold):
            return True

        return False


    def move_pose(self,
                  goal_pose_T: T,
                  viewer
                  ):
        self.i = 0

        
        q_goal = self.robot._eef_position_controller.ik.ik(goal_pose_T.matrix, self.robot.get_current_joint_state()[0])

        while 1:
            # get states
            q_current, _ = self.robot.get_current_joint_state()
            current_eef_pose_T = self.robot.get_eef_pose_in_base_frame()
            # current_joint_state = self.robot.get_current_joint_state()
            # measured_wrench = self.robot.get_fts_data(transform_to_base=True)
            

            # Check the termination condition
            if self.termination(q_current, q_goal):
                # self.data_eva.save()
                print("reached goal pose {}".format(q_current))
                break
            
            # EEF Position control
            next_eef_goal = self.robot._eef_position_controller.eef_position_control(
                current_eef_pose = current_eef_pose_T,
                target_eef_pose = goal_pose_T,
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
            
            self.i += 1
        
        self._mj_data.qpos[0:6] = self.robot._eef_position_controller.ik.ik(goal_pose_T.matrix, q_current)

    def set_gripper_position(self, position: float, viewer):
        """
        Set the gripper position (width between fingers).
        position: float, range from [0.0, 0.08] meter
        """
        min_width = 0.0181 # mujoco measured opening when logical position = 0.0
        max_width = 0.0983  # mujoco measured opening when logical position = 0.08
        threshold = 0.0009  # stop when measured opening is within this tolerance

        # map logical position in [0.0, 0.08] to mujoco measured opening
        position_clipped = float(np.clip(position, 0.0, 0.08))
        target_measured = min_width + (position_clipped / 0.08) * (max_width - min_width)

        # set actuator command (0..255), inverted in this model
        self._mj_data.ctrl[6] = 255 - int(position_clipped / 0.08 * 255)

        # step simulation until measured opening reaches the target within threshold
        print("[SET GRIPPER] Target opening: {:.4f} m".format(position))
        current = self.get_gripper_opening()
        while abs(current - target_measured) > threshold:
            self.robot.spin()
            self.step_mj_simulation()
            current = self.get_gripper_opening()
            print("[SET GRIPPER] Current opening: {:.4f} m, Target: {:.4f} m".format(current, target_measured))

            if viewer is not None:
                viewer.sync()
        
        # compute logical position (0..0.08) from mujoco measured opening
        measured_logical = (current - min_width) / (max_width - min_width) * 0.08
        measured_logical = float(np.clip(measured_logical, 0.0, 0.08))
        print("[SET GRIPPER] Position {:.4f} m, Reached {:.4f} m".format(position_clipped, measured_logical))

    
    def get_gripper_opening(self) -> float:
        """
        Get the gripper opening width in meter
        """
        left_finger_pos, _ = get_body_pose_in_world(self._mj_model, self._mj_data, "2f85_left_silicone_pad")
        right_finger_pos, _ = get_body_pose_in_world(self._mj_model, self._mj_data, "2f85_right_silicone_pad")
        opening_width = np.linalg.norm(left_finger_pos - right_finger_pos)

        return opening_width
    
    
    def exec_labit(self):
        """
        Main function to execute the LABIT benchmark task.
        """
        # self._mj_scene = mujoco.MjvScene(self._mj_model, maxgeom=150000)

        with mujoco.viewer.launch_passive(self._mj_model, self._mj_data, show_left_ui=False, show_right_ui=False) as viewer:
            
            self.update_view_scale()
            update_view_camera_parameter(viewer, view_type="labit_benchmark")
            self.update_view_opt(viewer)
            viewer.sync()
            
            # print_object_names(self._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, self._mj_model.nbody, "Bodies in the model")

            # relative movement
            # current_eef_pose_T = self.robot.get_eef_pose_in_base_frame()  
            # goal_pose_T = current_eef_pose_T
            # goal_pose_T.translation += np.array([0.0, 0.0, 0.12])

            # absolute goal pose
            # goal_pose_T = T(translation=np.array([0.7, 0.0, 0.4]),
            #                   quaternion=np.array([0.707, 0.707, 0.0, 0.0]))

            # self.set_gripper_position(0.01, viewer)  # open gripper

            # object to grasp
            goal_pose_T = get_relative_pose(self._mj_model, self._mj_data, "base", "positioning_pin_d5_20_1_body")
            goal_pose_T.translation[2] += 0.4

            self.move_pose(goal_pose_T, viewer)
            # q_init, _ = self.robot.get_current_joint_state()
            
            current_eef_pose_T = self.robot.get_eef_pose_in_base_frame()  
            goal_pose_T = current_eef_pose_T
            goal_pose_T.translation += np.array([0.0, 0.0, -0.08])
            self.move_pose(goal_pose_T, viewer)
            self.set_gripper_position(0.0045, viewer)

            while 1:
                # self._mj_data.qpos[0:6] = self.eef_position_controller.ik.ik(goal_pose_T.matrix, q_init)
                self.robot.spin()
                self.step_mj_simulation()
                
                if viewer != None:
                    viewer.sync()

            time.sleep(60)
            return


    def exec_insertion_headless(self):
        start_pose_T, goal_pose_T = self.get_fixed_start_and_goal_pose()

        # Move the robot to the initial position
        self.robot.move_to_eef_pose(
            viewer=None,
            eef_pose=start_pose_T.get_pos_quat_list(quat_format='xyzw'),
            qpos_thresh=0.001,
            executing=True
        )

        self.move_pose(goal_pose_T, None)

        # self.data_eva.plot_data()

        return


  

if __name__ == "__main__":
    
    task_env_config_path = "/workspace/qbit/configs/envs/ur5e_labit_benchmark.yaml"
    
    mj = PositionBasedInsertion(
        task_env_config_path=task_env_config_path,
        server_modus=True,
        sim_timestep=SIM_TIMESTEP,
        )
    mj.exec_labit()
