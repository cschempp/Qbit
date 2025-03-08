"""
Force-based insertion with admittance control:
 - Randomize the start pose with consideration of the perception error
 - Adapt the parameter for the admittance control
"""


import numpy as np
import mujoco
import mujoco.viewer

from qbit.controllers.eef_position_controller import EEFPositionController
from qbit.controllers.eef_admittance_controller import EEFAdmittanceController
from qbit.utils.tf_utils import T
from qbit.utils.mj_viewer_utils import update_view_camera_parameter
from qbit.sim_envs.mujoco_env_insertion import MjEnvInsertion

import keras
import cv2


NUM_RUNS = 300

RESULT_DIR = "/workspace/examples/experiment_results/force_based"

SIM_TIMESTEP = 0.001

NN_CONTROL_DT = 0.01
ADMITTANCE_CONTROL_T = 0.01
JOINT_POSITION_CONTROL_T = 0.001  # Second

POS_RANDOM_LIMIT = 0.002 # meter
ROT_RANDOM_LIMIT = 1.5 # degree


NN_MODEL_PATH = "/workspace/insertion/insertionnet/checkpoints/insertionnet_IF_sim_peg"

class LearningBasedInsertion(MjEnvInsertion):
    
    
    def __init__(self,
                 task_env_config_path: str,
                 sim_timestep: float = 0.001,
                 rendering_timestep: float = 0.033,
                 rt_factor: float = 0.0,
                 headless: bool = False,
                 server_modus: bool = True,
                 ):
        
        super(LearningBasedInsertion, self).__init__(
            task_env_config_path,
            sim_timestep,
            rendering_timestep,
            rt_factor,
            headless,
            server_modus,
        )
        
        self.z_insertion_depth = 0.073 # insertion depth
        
        self.insertion_goal_T = T(
            translation = [0.6, 0.0, 0.156],
            quaternion = [0.707, 0.707, 0.0, 0.0]
        )
        
        self.eef_position_controller = EEFPositionController(
            kp = 2,
            eef_pos_vel_max = np.array([0.1, 0.1, 0.1]) * 4
        )
        self.eef_admittance_controller = EEFAdmittanceController()
        
        if server_modus:
            self.renderer = mujoco.Renderer(self.robot._mj_model, 720, 1280)
            self.in_net = keras.saving.load_model(NN_MODEL_PATH,
                                                  custom_objects=None,
                                                  compile=True,
                                                  safe_mode=True)
    def get_insertionnet_action(self, wrench):
        self.renderer.update_scene(self.robot._mj_data, "ur_cam")
        img = self.renderer.render()

        img = cv2.resize(img[:, 500:], dsize=(256, 256))

        action = self.in_net([np.expand_dims(img, 0), np.expand_dims(wrench, 0)])
        
        return action.numpy()[0, :]
    


    def insertion(self,
                  goal_pose_T: T,
                  viewer
                  ):
        self.i = 0
        while 1:
            # print("*" * 50, self.i, "*" * 50)
            # 
            # get states
            current_eef_pose_T = self.robot.get_eef_pose_in_base_frame()
            current_joint_state = self.robot.get_current_joint_state()
            measured_wrench = self.robot.get_fts_data(transform_to_base=True)

            # Recoad data
            # self.data_eva.record(
            #     timestamp = self.i * ADMITTANCE_CONTROL_T,
            #     eef_fts=copy.deepcopy(measured_wrench),
            #     eef_pos=current_eef_pose_T.translation,
            #     eef_qua=current_eef_pose_T.quaternion,
            #     joint_states=current_joint_state
            # )
            
            # print(f"Measured wrench from FTS: {measured_wrench}")
            # print(f"Current EEF pos: {current_eef_pose_T.translation}")
            # print(f"Currnet EEF rot: {current_eef_pose_T.euler_rotation * 180 / np.pi}")

            # Check the termination condition
            if self.termination(current_eef_pose_T):
                # self.data_eva.save()
                return
            
            # EEF admittance control
            # get policy action
            action = self.get_insertionnet_action(measured_wrench)

            self.c = 1.0/800
            self.f_des = -10.0

            dz = - self.c * (self.f_des + measured_wrench[2])
            dz = np.clip(dz, -0.005, 0.005)
            rot = action[2:]
            
            # calculate the next goal pose
            rot_T = T.from_euler(translation=[action[0], action[1], dz],
                                 euler=rot)
            next_eef_goal_T =  current_eef_pose_T * rot_T
            
            # Joint position control
            self.robot.move_to_eef_pose(
                next_eef_goal_T.get_pos_quat_list(quat_format='xyzw'),
                qpos_thresh=0.001,
                executing=False
            )
            for _ in range(10):
                
                self.robot.spin()
                self.step_mj_simulation()
            viewer.sync()

            self.i += 1
        
    
    def exec_insertion(self):
        """
        Main function to execute the insertion task
        """
        
        with mujoco.viewer.launch_passive(self._mj_model, self._mj_data) as viewer:
            
            self.update_view_scale()
            update_view_camera_parameter(viewer)
            # self.update_view_opt(viewer)
            viewer.sync()
            
            start_pose_T, goal_pose_T = self.get_fixed_start_and_goal_pose()

            # Move the robot to the initial position
            self.robot.move_to_eef_pose(
                start_pose_T.get_pos_quat_list(quat_format='xyzw'),
                qpos_thresh=0.001,
                executing=True
            )
            viewer.sync()

            self.insertion(goal_pose_T, viewer)
            return
        
        

if __name__ == "__main__":
    
    task_env_config_path = "/workspace/qbit/configs/envs/ur5e_peg_task.yaml"
    
    mj = LearningBasedInsertion(
        task_env_config_path=task_env_config_path,
        server_modus=True
        )
    mj.exec_insertion()
