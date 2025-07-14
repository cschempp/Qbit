import mujoco_viewer
import mujoco
import numpy as np

from qbit.utils.tf_utils import T
from examples.position_based_insertion_pih import PositionBasedInsertion



NUM_RUNS = 300

RESULT_DIR = "/workspace/examples/experiment_results/position_based/exp1_decomposed_mesh"

SIM_TIMESTEP = 0.001

NN_CONTROL_DT = 0.01
ADMITTANCE_CONTROL_T = 0.01
JOINT_POSITION_CONTROL_T = 0.001  # Second

POS_RANDOM_LIMIT = 0.001 # meter
ROT_RANDOM_LIMIT = 1.5 # degree



class PositionBasedInsertionFTPlot(PositionBasedInsertion):
    def __init__(self, task_env_config_path, sim_timestep = 0.001, rendering_timestep = 0.033, rt_factor = 0, headless = True, server_modus = False):
        super().__init__(task_env_config_path, sim_timestep, rendering_timestep, rt_factor, headless, server_modus)
    
    def exec_insertion(self):
        """
        Main function to execute the insertion task
        """
    
        viewer = mujoco_viewer.MujocoViewer(model=self._mj_model, data=self._mj_data)
        viewer.add_line_to_fig(line_name="force_x", fig_idx=0)
        viewer.add_line_to_fig(line_name="force_y", fig_idx=0)
        viewer.add_line_to_fig(line_name="force_z", fig_idx=0)
        viewer.add_line_to_fig(line_name="torque_x", fig_idx=1)
        viewer.add_line_to_fig(line_name="torque_y", fig_idx=1)
        viewer.add_line_to_fig(line_name="torque_z", fig_idx=1)

        fig = viewer.figs[0]
        fig.flg_extend = 0
        #x range
        fig.range[0][0] = -50
        fig.range[0][1] = 0
        # y range
        fig.range[1][0] = -100
        fig.range[1][1] = 100
        fig.gridsize[0] = 5
        fig.gridsize[1] = 5

        fig = viewer.figs[1]
        fig.flg_extend = 0
        # x range
        fig.range[0][0] = -50
        fig.range[0][1] = 0
        # y range
        fig.range[1][0] = -50
        fig.range[1][1] = 50
        fig.gridsize[0] = 5 
        fig.gridsize[1] = 5

        self.update_view_scale()
        # update_view_camera_parameter(viewer)
        # viewer.sync()
        viewer.render()
        
        start_pose_T, goal_pose_T = self.get_fixed_start_and_goal_pose()
        print(start_pose_T)
        print(goal_pose_T)
        print("++++++++++++++++++++++++++++++++++++++")

        for _ in range(50):
            translation_offset = np.random.rand(3) * 0.004 - 0.002
            # translation_offset = np.random.rand(3) * 0.02 - 0.01
            # start_pose_T.translation += translation_offset
            # goal_pose_T.translation += translation_offset

            # set initial position directly
            q_goal = self.robot.get_q_goal(eef_pose=start_pose_T.get_pos_quat_list(quat_format='xyzw'))
            self._mj_data.qpos[0:6] = q_goal

            mujoco.mj_step(self._mj_model, self._mj_data)
            # viewer.sync()
            viewer.render()
            print("reached start pose")

            self.insertion(goal_pose_T, viewer)
        
        return
    
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

            viewer.add_data_to_line(line_name="force_x", line_data=measured_wrench[0], fig_idx=0)
            viewer.add_data_to_line(line_name="force_y", line_data=measured_wrench[1], fig_idx=0)
            viewer.add_data_to_line(line_name="force_z", line_data=measured_wrench[2], fig_idx=0)

            viewer.add_data_to_line(line_name="torque_x", line_data=measured_wrench[3], fig_idx=1)
            viewer.add_data_to_line(line_name="torque_y", line_data=measured_wrench[4], fig_idx=1)
            viewer.add_data_to_line(line_name="torque_z", line_data=measured_wrench[5], fig_idx=1)

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

            if viewer != None:    
                viewer.render()
            
            self.i += 1


if __name__ == "__main__":
    task_env_config_path = "/workspace/qbit/configs/envs/ur5e_pih_task.yaml"
    
    mj = PositionBasedInsertionFTPlot(
        task_env_config_path=task_env_config_path,
        server_modus=True,
        sim_timestep=SIM_TIMESTEP,
        )
    mj.exec_insertion()