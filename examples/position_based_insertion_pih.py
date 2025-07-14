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


from qbit.controllers.eef_position_controller import EEFPositionController
from qbit.utils.tf_utils import T
from qbit.utils.mj_viewer_utils import update_view_camera_parameter
from qbit.sim_envs.mujoco_env_insertion import MjEnvInsertion


NUM_RUNS = 300

RESULT_DIR = "/workspace/examples/experiment_results/position_based/exp1_decomposed_mesh"

SIM_TIMESTEP = 0.001

NN_CONTROL_DT = 0.01
ADMITTANCE_CONTROL_T = 0.01
JOINT_POSITION_CONTROL_T = 0.001  # Second

POS_RANDOM_LIMIT = 0.001 # meter
ROT_RANDOM_LIMIT = 1.5 # degree



class PositionBasedInsertion(MjEnvInsertion):
    
    
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
        
        # self.insertion_goal_T = T(
        #     translation = [0.6, 0.0, 0.115], #[0.6, 0.0, 0.15],
        #     quaternion = [0.707, 0.707, 0.0, 0.0]
        # )
        
        self.eef_position_controller = EEFPositionController(
            kp = 2,
            eef_pos_vel_max = np.array([0.1, 0.1, 0.1]) * 4
        )
        
        #self.data_eva = DataRecording(task_env_config_path=task_env_config_path)


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
            
            # get states
            current_eef_pose_T = self.robot.get_eef_pose_in_base_frame()
            current_joint_state = self.robot.get_current_joint_state()
            measured_wrench = self.robot.get_fts_data(transform_to_base=True)

            measured_wrench = self.robot.get_fts_data(transform_to_base=True)
            

            # Recoad data
            # self.data_eva.record(
            #     timestamp = self.i * ADMITTANCE_CONTROL_T,
            #     eef_fts=copy.deepcopy(measured_wrench),
            #     eef_pos=current_eef_pose_T.translation,
            #     eef_qua=current_eef_pose_T.quaternion,
            #     joint_states=current_joint_state
            # )
            
            # print("Iteration: ", self.i)
            # print(current_eef_pose_T.translation[2])
            # print("Iteration: ", self.i, " Measured wrench from FTS: ", measured_wrench[:3])
            # print(f"Measured wrench from FTS: {measured_wrench}")
            # print(f"Current EEF pos: {current_eef_pose_T.translation}")
            # print(f"Currnet EEF rot: {current_eef_pose_T.euler_rotation * 180 / np.pi}")

            # Check the termination condition
            if self.termination(current_eef_pose_T):
                # self.data_eva.save()
                return
            
            # EEF Position control
            next_eef_goal = self.eef_position_controller.eef_position_control(
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
                qpos_thresh=0.5 * np.pi/180,
                executing=False
            )

            for _ in range(1):
                self.robot.spin()
                self.step_mj_simulation()

                # id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "peg_body")
                # peg_pos = self._mj_data.xpos[id, :]
                # print(peg_pos[2])

            if viewer != None:    
                viewer.sync()
            
            self.i += 1
        
    
    def exec_insertion(self):
        """
        Main function to execute the insertion task
        """
        
        with mujoco.viewer.launch_passive(self._mj_model, self._mj_data, show_left_ui=False, show_right_ui=False) as viewer:
            
            self.update_view_scale()
            update_view_camera_parameter(viewer)
            # self.update_view_opt(viewer)
            viewer.sync()
            
            start_pose_T, goal_pose_T = self.get_fixed_start_and_goal_pose()
            print(start_pose_T)
            print(goal_pose_T)
            print("++++++++++++++++++++++++++++++++++++++")
            # Move the robot to the initial position
            # self.robot.move_to_eef_pose(
            #     viewer=viewer,
            #     eef_pose=start_pose_T.get_pos_quat_list(quat_format='xyzw'),
            #     qpos_thresh=0.01 * np.pi/180,
            #     executing=True,
            # )

            for _ in range(50):
                start_pose_T.translation += np.random.rand(3) * 0.004 - 0.002
                # set initial position directly
                q_goal = self.robot.get_q_goal(eef_pose=start_pose_T.get_pos_quat_list(quat_format='xyzw'))
                self._mj_data.qpos[0:6] = q_goal
                mujoco.mj_step(self._mj_model, self._mj_data)
                viewer.sync()
                print("reached start pose")

                self.insertion(goal_pose_T, viewer)
                # self.data_eva.plot_data()
            
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

        self.insertion(goal_pose_T, None)

        # self.data_eva.plot_data()

        return


  

if __name__ == "__main__":
    
    task_env_config_path = "/workspace/qbit/configs/envs/ur5e_pih_task.yaml"
    
    mj = PositionBasedInsertion(
        task_env_config_path=task_env_config_path,
        server_modus=True,
        sim_timestep=SIM_TIMESTEP,
        )
    mj.exec_insertion()