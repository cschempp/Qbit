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


NUM_RUNS = 300

RESULT_DIR = "/workspace/examples/experiment_results/force_based"

SIM_TIMESTEP = 0.001

NN_CONTROL_DT = 0.01
ADMITTANCE_CONTROL_T = 0.01
JOINT_POSITION_CONTROL_T = 0.001  # Second

POS_RANDOM_LIMIT = 0.002 # meter
ROT_RANDOM_LIMIT = 1.5 # degree



class ForceBasedInsertion(MjEnvInsertion):
    
    
    def __init__(self,
                 task_env_config_path: str,
                 sim_timestep: float = 0.001,
                 rendering_timestep: float = 0.033,
                 rt_factor: float = 0.0,
                 headless: bool = False,
                 server_modus: bool = False,
                 ):
        
        super(ForceBasedInsertion, self).__init__(
            task_env_config_path,
            sim_timestep,
            rendering_timestep,
            rt_factor,
            headless,
            server_modus,
        )
        
        self.insertion_goal_T = T(
            translation = [0.6, 0.0, 0.156],
            quaternion = [0.707, 0.707, 0.0, 0.0]
        )
        
        self.eef_position_controller = EEFPositionController(
            kp = 2,
            eef_pos_vel_max = np.array([0.1, 0.1, 0.1]) * 4
        )
        self.eef_admittance_controller = EEFAdmittanceController()
            
        
    def termination_data_collection(self,
                                    current_eef_pose_T: T,
                                    measured_wrench: np.ndarray,
                                    insertion_step: int,):
        if any([
            current_eef_pose_T.translation[2] < self.insertion_goal_T.translation[2] + 0.01,
            np.max(np.abs(measured_wrench)) > 20,
            insertion_step > 1000
        ]):
            self.run_id += 1
            self.data_id = 0
            return True
        else:
            return False


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
            next_eef_goal_T = self.eef_admittance_controller.admittance_control(
                actual_wrench=measured_wrench,
                current_eef_pose=current_eef_pose_T,
            )
            
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
    
    mj = ForceBasedInsertion(
        task_env_config_path=task_env_config_path,
        server_modus=True
        )
    mj.exec_insertion()
