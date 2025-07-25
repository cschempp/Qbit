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
import copy
import glob
import os

from qbit.controllers.eef_position_controller import EEFPositionController
from qbit.utils.tf_utils import T
from qbit.utils.mj_viewer_utils import update_view_camera_parameter
from qbit.utils.data_recording_utils import DataRecording
from qbit.sim_envs.mujoco_env_deformation import MjEnvDeformation
from qbit.sim_envs.mujoco_env_base import MujocoEnvBase
from qbit.utils.mesh_processing import MeshObjects

import yaml

NUM_RUNS = 300

RESULT_DIR = "/workspace/examples/experiment_results/position_based/exp_pipe_deformation"

SIM_TIMESTEP = 1e-8

NN_CONTROL_DT = 0.01
ADMITTANCE_CONTROL_T = 0.01

JOINT_POSITION_CONTROL_T = 0.001  # Second

POS_RANDOM_LIMIT = 0.0 # meter
ROT_RANDOM_LIMIT = 0.0 # degree



class PositionBasedDeformation(MjEnvDeformation):
    
    
    def __init__(self,
                 task_env_config_path: str,
                 sim_timestep: float = 0.001,
                 rendering_timestep: float = 0.033,
                 rt_factor: float = 0.0,
                 headless: bool = True,
                 server_modus: bool = False,
                 ):
        
        super(PositionBasedDeformation, self).__init__(
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
        
        self.data_eva = DataRecording(task_env_config_path=task_env_config_path)


    def termination(self, 
                    current_eef_pose_T: T,
                    threshold: float = 0.0
                    ) -> bool:
        
        if current_eef_pose_T.translation[2] <= self.insertion_goal_T.translation[2] + threshold:
            return True
        if self.i >= 500:
            return True
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

            # Recoad data
            self.data_eva.record(
                timestamp = self.i * ADMITTANCE_CONTROL_T,
                eef_fts=copy.deepcopy(measured_wrench),
                eef_pos=current_eef_pose_T.translation,
                eef_qua=current_eef_pose_T.quaternion,
                joint_states=current_joint_state
            )
            
            # print("Iteration: ", self.i)
            # print(current_eef_pose_T.translation[2])
            # print("Iteration: ", self.i, " Measured wrench from FTS: ", measured_wrench[:3])
            # print(f"Measured wrench from FTS: {measured_wrench}")
            # print(f"Current EEF pos: {current_eef_pose_T.translation}")
            # print(f"Currnet EEF rot: {current_eef_pose_T.euler_rotation * 180 / np.pi}")

            # Check the termination condition
            if self.termination(current_eef_pose_T):
                self.data_eva.save()
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
            # set initial position directly
            q_goal = self.robot.get_q_goal(eef_pose=start_pose_T.get_pos_quat_list(quat_format='xyzw'))
            self._mj_data.qpos[0:6] = q_goal
            mujoco.mj_step(self._mj_model, self._mj_data)
            viewer.sync()

            self.insertion(goal_pose_T, viewer)
            self.data_eva.plot_data()
            
            return


    def exec_insertion_headless(self):
        start_pose_T, goal_pose_T = self.get_fixed_start_and_goal_pose()

        # Move the robot to the initial position
        # self.robot.move_to_eef_pose(
        #     eef_pose=start_pose_T.get_pos_quat_list(quat_format='xyzw'),
        #     qpos_thresh=0.001,
        #     executing=True
        # )

        q_goal = self.robot.get_q_goal(eef_pose=start_pose_T.get_pos_quat_list(quat_format='xyzw'))
        self._mj_data.qpos[0:6] = q_goal
        mujoco.mj_step(self._mj_model, self._mj_data)
        
        self.insertion(goal_pose_T, None)

        self.data_eva.plot_data()

        return



if __name__ == "__main__":
    
    # task_env_config_path = "/workspace/qbit/configs/envs/ur5e_deformation_task.yaml"
    
    # primitives_paths = glob.glob(os.path.join("qbit", "assets", "task_env", "primitives", "*"))
    # materials = ["steel", "plastic", "wood", "rubber"]
  
    # # task_env_config_path_ = "/workspace/qbit/configs/envs/data.yaml"
    # mj = PositionBasedDeformation(
    #     task_env_config_path=task_env_config_path,
    #     server_modus=True,
    #     sim_timestep=SIM_TIMESTEP
    #     )
    # mj.exec_insertion()

    task_env_config_path = "/workspace/qbit/configs/envs/ur5e_deformation_task.yaml"
    
    primitives_paths = glob.glob(os.path.join("qbit", "assets", "task_env", "primitives", "*"))
    materials = ["steel", "plastic", "wood", "rubber"]
    RESULT_DIR = "/workspace/examples/experiment_results/position_based/exp_pipe_deformation"

    for prim_path in primitives_paths:
        for material_male in materials:
            for material_female in materials:
                
                config = MujocoEnvBase.parse_qbit_config_yaml(task_env_config_path)
                prim_name = prim_path.split(os.sep)[-1]
                prim_type = prim_name.split("_")[0]
                mesh_path_peg = os.path.join(os.sep,"workspace", prim_path, prim_name + "_male.stl")
                mesh_path_hole = os.path.join(os.sep,"workspace", prim_path, prim_name + "_male.stl")

                nzfilename = prim_type + "_" + prim_name + "_" + "male" + "_" + material_male + "_" + material_female + ".nc"
                if os.path.exists(os.path.join(RESULT_DIR, prim_name, nzfilename)):
                    print(nzfilename + " already simualted. skipping.")
                    continue

                # mo = MeshObjects(os.path.join(os.sep, "workspace", prim_path, prim_name + "_female.stl"))
                # mo.decomposition_with_coacd(threshold=0.01)

                # hole
                config["task_objects"][1]["obj_name"] = prim_name + "_female"
                config["task_objects"][1]["obj_type"] = prim_type
                config["task_objects"][1]["mesh_path"] = mesh_path_hole
                config["task_objects"][1]["material"] = material_female
                config["task_objects"][1]["scale"] = [0.001, 0.001, 0.001]
                # peg
                config["task_objects"][0]["obj_name"] = prim_name + "_male"
                config["task_objects"][0]["obj_type"] = prim_type
                config["task_objects"][0]["mesh_path"] = mesh_path_peg
                config["task_objects"][0]["material"] = material_male

                with open('/workspace/qbit/configs/envs/data.yaml', 'w') as outfile:
                    yaml.dump(config, outfile, default_flow_style=False)

                task_env_config_path_ = "/workspace/qbit/configs/envs/data.yaml"
                mj = PositionBasedDeformation(
                    task_env_config_path=task_env_config_path_,
                    server_modus=True,
                    sim_timestep=SIM_TIMESTEP
                    )
                mj.exec_insertion_headless()