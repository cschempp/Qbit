"""
Task object base class
"""

import os
import glob
from typing import Tuple, Literal
from loguru import logger

import numpy as np

import mujoco
import mujoco.viewer

from qbit.utils.tf_utils import T
from qbit.utils.mujoco_utils import convert_quat_to_xyzw



class DecomposedObject:
    
    def __init__(self,
                 mj_spec,
                 config_dict: dict):
        
        self._mj_spec = mj_spec
        self._config = config_dict
        
        self.load_objects(self._config)
        
    def load_objects(self, config):
        
        mesh_path = config.get('mesh_path')
        
        if self._config.get('attach_body') == 'world':
            obj_body = self._mj_spec.worldbody.add_body(
                name = f"{config['obj_name']}_body",
                pos = config.get('attach_pose')['position'],
                mass = config.get('mass'),
                )
        else:
            parent_body_name = config.get('attach_body')
            obj_body = self._mj_spec.find_body(parent_body_name).add_body(
                name = f"{config.get('obj_name')}_body",
                pos = config.get('attach_pose')['position'],
                mass = config.get('mass'),
                )

        # load the mesh files
        mesh_files = sorted(glob.glob(os.path.join(mesh_path, "*.obj")))
        mesh_color = config.get('mesh_color', [1, 0, 0, 1]),
        for i, f in enumerate(mesh_files):
            # mesh_color = [0, 0, 1, 1]
            # mesh_color = np.random.rand(3).tolist() + [1.0]  # Random RGB color with alpha = 1.0
            geom = obj_body.add_geom(
                type = mujoco.mjtGeom.mjGEOM_MESH,
                meshname = f"{config.get('obj_name')}_mesh_{i}",
                quat = config.get('attach_pose')['quaternion'],
                rgba = mesh_color[0]
            )
            mesh = self._mj_spec.add_mesh()
            mesh.name = f"{config.get('obj_name')}_mesh_{i}"
            mesh.file = f
            mesh.scale = [0.001, 0.001, 0.001]

        print("loaded mesh files")



class MeshObject:
    
    def __init__(self,
                 mj_spec,
                 config_dict: dict):
        
        self._mj_spec = mj_spec
        self._config = config_dict
        
        self.load_mesh_object(self._config)
        
    def load_mesh_object(self, config):
        
        if self._config.get('attach_body') == 'world':
            obj_body = self._mj_spec.worldbody.add_body(
                name = f"{config['obj_name']}_body",
                pos = config.get('attach_pose')['position'],
                )
        else:
            parent_body_name = config.get('attach_body')
            obj_body = self._mj_spec.find_body(parent_body_name).add_body(
                name = f"{config.get('obj_name')}_body",
                pos = config.get('attach_pose')['position'],
                quat = config.get('attach_pose')['quaternion'],
                )

        # load the mesh files
        obj_body.add_geom(
            type = mujoco.mjtGeom.mjGEOM_MESH,
            meshname = f"{config.get('obj_name')}_mesh",
            condim = config.get('contact').get('condim', 3),
            rgba = config.get('mesh_color', [1, 0, 0, 1]),
            mass = config.get('mass'), 
        )
        
        mesh = self._mj_spec.add_mesh()
        mesh.name = f"{config.get('obj_name')}_mesh"
        mesh.file = config.get('mesh_path')
        mesh.scale = config.get('scale')

        print(f"loaded object {config.get('obj_name')}")


class BuildInObject:
    """
    Add the built-in object (primitive shapes) to the MuJoCo model
    """

    def __init__(self,
                 mj_spec):
        self._mj_spec = mj_spec


    def add_box(self, 
                pose: T,
                box_size: Tuple[float, float, float],
                obj_name = 'box-1'):
        
        body = self._mj_spec.worldbody.add_body(
            name = obj_name,
            pos = pose.translation,
            quat = convert_quat_to_xyzw(pose.quaternion),
            mass = 1.0,
        )
        geom = body.add_geom(
            type = mujoco.mjtGeom.mjGEOM_BOX,
            size = box_size,
            density = 1000,
            rgba = [0, 0, 1, 0.5],
            condim = 3,
        )
        geom.friction[0] = 0.1
        return
