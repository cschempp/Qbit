"""
Using the robot xml as base to create the simulation environment.
 - Adapt the physics parameters and solver parameters
 - change the lighting and camera settings
"""

import time
import yaml
from typing import List

import mujoco
import mujoco.viewer

from qbit.robots.ur5e_mj import UR5eMjArm
from qbit.robots.kuka_iiwa14_mj import KUKAiiwa14MjArm
from qbit.objects.object_base import DecomposedObject, MeshObject, FlexcompObject
from qbit.objects.env_objects import MjEnvObjects
from qbit.utils.mj_viewer_utils import update_view_camera_parameter
from qbit.interfaces.grpc.mj_grpc_proxy import QbitMjGrpcProxy


AVAILABLE_COMPONENTS = {
    'ur5e': UR5eMjArm,
    'iiwa14': KUKAiiwa14MjArm,
}


class MujocoEnvBase:

    
    def __init__(self,
                 task_env_config_path: str,
                 sim_timestep: float = 0.001,
                 rendering_timestep: float = 0.033,
                 rt_factor: float = 0.0,
                 headless: bool = False,
                 server_modus: bool = True,
                 ):
        """
        In client_modus, the simulation will be run in server mode and can 
        be controlled by the client through the interface.
        """
        
        self._sim_timestep = sim_timestep
        self._rendering_timestep = rendering_timestep
        self._rt_factor = rt_factor
        self._headless = headless
        self._sim_time = 0
        self._last_render_time = 0

        
        self.load_env(task_env_config_path)
        
        self.compile_model()
        
        self.reset_sim()
        
        self._server_modus = server_modus
        if server_modus:
            self._interface = QbitMjGrpcProxy()
            self._interface.start()

    
    def reset_sim(self):
        """
        Reset the simulation
        """
        self.robot.reset()
    
    
    def load_env(self, task_env_config_path):
        
        self._config = self.parse_qbit_config_yaml(task_env_config_path)
        
        # load the robot
        self._mj_spec = self.load_robot(
            self._config.get('robot'),
        )
        
        # load environment objects
        self.load_env_objects(self._config.get('env_objects'))

        # load task objects
        start_position_hole, insertion_depth = self.load_task_objects(self._config.get('task_objects'))

        return start_position_hole, insertion_depth
        
        

    def load_robot(self,
                   env_robot_config: dict):
        """
        Load the robot xml file and return the _mj_spec object
        Return the _mj_spec object for adding the environment and task objects.
        """
        
        robot_config = self.parse_qbit_config_yaml(env_robot_config.get('robot_config_path'))
        
        print(f"Loading robot from Config: {robot_config}")

        robot_name = robot_config.get('robot_type')        
        if robot_name not in AVAILABLE_COMPONENTS:
            raise ValueError(f"Robot class {robot_name} is not supported.")

        # load the robot
        self.robot = AVAILABLE_COMPONENTS[robot_name](
            qbit_robot_config = robot_config,
            headless = self._headless,
            arm_base_pos = env_robot_config.get('base_pose')['position'],
            arm_base_qua = env_robot_config.get('base_pose')['quaternion'], 
            )
        return self.robot.mj_spec


    def load_env_objects(self, env_objects: List[dict]):
        """
        Load the environment
        """
        self.env_objects = MjEnvObjects(self._mj_spec, env_objects)


    def load_task_objects(self, task_objects: list):
        """
        Load the task objects
        """

        friction_list = {
            "steel": {
                "steel": 0.4,
                "plastic": 0.2,
                "wood": 0.5,
                "rubber": 0.2,
            },
            "plastic": {
                "steel": 0.2,
                "plastic": 0.25,
                "wood": 0.3,
                "rubber": 0.6,
            },
            "wood": {
                "steel": 0.5,
                "plastic": 0.3,
                "wood": 0.5,
                "rubber": 0.9,
            },
            "rubber": {
                "steel": 0.2,
                "plastic": 0.6,
                "wood": 0.9,
                "rubber": 1.8,
            }
        }

        self.materials = [task_obj.get('material') for task_obj in task_objects]
        self.friction = friction_list[self.materials[0]][self.materials[1]]
        self.task_objects = task_objects

        for task_obj in task_objects:
            if task_obj["obj_name"] == "hole":
                if task_obj.get('mesh_type') in ['coacd', 'vhacd']:
                    _object = DecomposedObject(self._mj_spec, task_obj, self.friction)
                elif task_obj.get('mesh_type') in ['mesh']:
                    _object = MeshObject(self._mj_spec, task_obj, self.friction)
                elif task_obj.get('mesh_type') in ['flexcomp']:
                    _object = FlexcompObject(self._mj_spec, task_obj, self.friction)
            else:
                if task_obj.get('mesh_type') in ['coacd', 'vhacd']:
                    DecomposedObject(self._mj_spec, task_obj, self.friction)
                elif task_obj.get('mesh_type') in ['mesh']:
                    MeshObject(self._mj_spec, task_obj, self.friction)
                elif task_obj.get('mesh_type') in ['flexcomp']:
                    FlexcompObject(self._mj_spec, task_obj, self.friction)

        return _object.start_position_hole, _object.insertion_depth
    
    def compile_model(self):
        """
        Compile the model and share the mj_model with 
        all active components such as the robot arm, gripper, and task objects.
        """
        self._mj_model = self._mj_spec.compile()
        self._mj_model.opt.timestep = self._sim_timestep
        self._mj_data = mujoco.MjData(self._mj_model)
        
        print("Compiled the model")
        
        self.robot.update_mj_pointer(self._mj_model, self._mj_data)
        
        return self._mj_model
    
    
    def setup(self):
        return NotImplementedError

    
    def rec_action(self):
        """
        Receive action from the interface
        For server modus, the action will be received from the interface
        """
        is_new_cmd, cmd = self._interface.check_rec_cmd_buffer()
        if not is_new_cmd:
            return
        else:
            print("Received new command", cmd)
            
            if cmd['cmd_type'] == 'joint_pos':
                self.robot.update_goal_joint_pos(cmd['cmd_data'])
            if cmd['cmd_type'] == 'eef_pose':
                self.robot.move_to_eef_pose(cmd['cmd_data'])
            return
        
            
    def update_interface_state_buffer(self):
        """
        Update the interface state buffer
        """
        robot_state = self.robot.get_current_joint_state()
        self._interface.update_robot_state_buffer(robot_state)
        return

    
    def rec_task(self):
        """
        Receive task: randomize the parameters
        """
        return NotImplementedError

    
    def rendering(self, viewer):
        if self._sim_time - self._last_render_time > self._rendering_timestep:
            viewer.sync()
            self._last_render_time = self._sim_time

        
    @staticmethod
    def parse_qbit_config_yaml(yaml_file_path):
        """
        Parse the qbit config yaml file
        """
        with open(yaml_file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        return config


    def update_view_opt(self, viewer):
        """
        Update the viewer options
            - show the site frame for debugging
            - https://mujoco.readthedocs.io/en/3.2.7/APIreference/APItypes.html#mjvoption
        """
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
        # https://mujoco.readthedocs.io/en/3.2.7/APIreference/APItypes.html#mjtvisflag
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1


    def update_view_scale(self):
        self._mj_model.vis.scale.contactwidth = 0.01
        self._mj_model.vis.scale.contactheight = 0.01
        self._mj_model.vis.scale.forcewidth = 0.01
        self._mj_model.vis.map.force = 0.00


    def step_mj_simulation(self):
        """
        Step the simulation
        """
        mujoco.mj_step(self._mj_model, self._mj_data)

    
    def spin(self, steps = 10, viewer = None):
        """
        For testing the simulation in the development mode
        """
        for i in range(steps):
            self.robot.spin()
            self.step_mj_simulation()
            self._sim_time += self._sim_timestep
            if viewer:
                viewer.sync()
            
    def spin_with_viewer(self):
        """
        Spin the simulation
        """
        with mujoco.viewer.launch_passive(self._mj_model, self._mj_data) as viewer:
            
            self.update_view_scale()
            update_view_camera_parameter(viewer)
            # self.update_view_opt(viewer)
            viewer.sync()
            
            while True:
                
                start_t = time.time()
                
                # check the received buffer
                if self._server_modus:
                    self.rec_action()

                self.robot.spin()
                
                self.step_mj_simulation()
                self._sim_time += self._sim_timestep
                
                self.update_interface_state_buffer()
                
                self.rendering(viewer)
                
                end_t = time.time()

                # sleep, if the simulation is faster than the desired real-time factor
                if self._rt_factor > 0:
                    sleep_time = self._sim_timestep * (1 / self._rt_factor) - (end_t - start_t)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        # print(f"Sleeping for {sleep_time * 1000} ms "
                        #       f"Max RT-Factor: {(sleep_time/(end_t-start_t)) + 1}")
                    else:
                        pass
                        # print(f"Execution exceeds desired period. "
                        #       f"Loop time: {(end_t - start_t) * 1000 } ms")


if __name__ == "__main__":
    
    task_env_config_path = "qbit/configs/envs/ur5e_peg_task.yaml"
    
    mj = MujocoEnvBase(
        task_env_config_path=task_env_config_path
        )
    mj.spin_with_viewer()
