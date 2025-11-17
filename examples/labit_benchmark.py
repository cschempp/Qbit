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
from qbit.utils.mujoco_utils import get_relative_pose, get_body_pose_in_world
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
                    current_eef_pose_T: T,
                    goal_pose_T: T,
                    threshold: float = 0.0005
                    ) -> bool:

        # if current_eef_pose_T.translation[2] < goal_pose_T.translation[2] + threshold:
        #      return True

        error = np.linalg.norm(current_eef_pose_T.translation - goal_pose_T.translation)
        print("current error: {}".format(error))
        if  error < threshold:
            return True
        if self.i >= 1000:
             return True
        return False


    def move_pose(self,
                  goal_pose_T: T,
                  viewer
                  ):
        self.i = 0

        while 1:
            # get states
            current_eef_pose_T = self.robot.get_eef_pose_in_base_frame()
            current_joint_state = self.robot.get_current_joint_state()
            measured_wrench = self.robot.get_fts_data(transform_to_base=True)
            
            # Check the termination condition
            if self.termination(current_eef_pose_T, goal_pose_T):
                # self.data_eva.save()
                print("reached goal pose {}".format(current_eef_pose_T))
                return
            
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

            # relative movement
            current_eef_pose_T = self.robot.get_eef_pose_in_base_frame()  
            # goal_pose_T = current_eef_pose_T
            # goal_pose_T.translation += np.array([0.0, 0.0, 0.12])

            # absolute goal pose
            # goal_pose_T = T(translation=np.array([0.7, 0.0, 0.4]),
            #                   quaternion=np.array([0.707, 0.707, 0.0, 0.0]))

            self._mj_data.ctrl[6] = 200  # close the gripper [0, 255] [open, close]
            # for _ in range(500):
            #     # self.robot.spin()
            #     self.step_mj_simulation()
            #     viewer.sync()

            # object to grasp
            goal_pose_T = get_relative_pose(self._mj_model, self._mj_data, "base", "positioning_pin_d5_20_1_body")
            goal_pose_T.translation[2] += 0.3

            self.move_pose(goal_pose_T, viewer)
            # q_init, _ = self.robot.get_current_joint_state()
            
            current_eef_pose_T = self.robot.get_eef_pose_in_base_frame()  
            goal_pose_T = current_eef_pose_T
            goal_pose_T.translation += np.array([0.0, 0.0, -0.04])
            self.move_pose(goal_pose_T, viewer)

            self._mj_data.ctrl[6] = 245
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
