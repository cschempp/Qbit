"""
Task object base class
"""

import gmsh
import os
import glob
from typing import Tuple, Literal
from loguru import logger
import xml.etree.ElementTree as ET

import mujoco._specs
import numpy as np

import trimesh

import mujoco
import mujoco.viewer

from qbit.utils.tf_utils import T
from qbit.utils.mujoco_utils import convert_quat_to_xyzw
from qbit.utils.mesh_processing import MeshObjects


material_list = {
    "steel": {
        "solref": [-500, -0.1], #-0.1
        "density": 7850, 
    },
    "plastic": {
        "solref": [-10.0, -0.1], #-0.01
        "density": 1190, 
    },
    "wood": {
        "solref": [-5.0, -0.1], #-0.01
        "density": 700, 
    },
    "rubber": {
        "solref": [-1.0, -0.1],
        "density": 920, 
    },
}


class BaseObject:
    def __init__(self, 
                 mj_spec, 
                 config_dict: dict,
                 friction: float):
        
        self._mj_spec = mj_spec
        self._config = config_dict
        self.friction = friction

        self.start_position_hole, self.insertion_depth = self.get_hole_pose_depth(self._config)
        self.attach_body(config=self._config)

    def get_hole_pose_depth(self, config):
        self.mesh_path = config.get('mesh_path')
        
        meshfile = os.path.join(*self.mesh_path.split(os.sep)[2:-1],self.mesh_path.split(os.sep)[-2] + "_female.stl")
        mesh = trimesh.load_mesh(meshfile)
        insertion_depth = mesh.extents[2] * 0.001
        start_position_hole = np.array(config.get('attach_pose')['position']) + np.array([0, 0, insertion_depth/2])
        
        return start_position_hole, insertion_depth
    
    def attach_body(self, config):
        if self._config.get('attach_body') == 'world':
            self.obj_body = self._mj_spec.worldbody.add_body(
                name = f"{config['obj_name']}_body",
                pos = self.start_position_hole,
                quat = config.get('attach_pose')['quaternion'],
                )
        else:
            parent_body_name = config.get('attach_body')
            self.obj_body = self._mj_spec.find_body(parent_body_name).add_body(
                name = f"{config.get('obj_name')}_body",
                pos = config.get('attach_pose')['position'],
                quat = config.get('attach_pose')['quaternion'],
                )


class DecomposedObject(BaseObject):
    
    def __init__(self,
                 mj_spec,
                 config_dict: dict,
                 friction: float):
        super(DecomposedObject, self).__init__(mj_spec, config_dict, friction)

        _mp = MeshObjects(obj_path=self._config.get('mesh_path'))
        _mp.decomposition_with_coacd(threshold=0.02)
        self._decomposed_mesh_dir = _mp._decomposed_mesh_dir

        self.load_decomposed_object(config=self._config)

    def load_decomposed_object(self, config):
        
        # load the mesh files
        mesh_files = sorted(glob.glob(os.path.join(self._decomposed_mesh_dir, "*.obj")))
        mesh_color = config.get('mesh_color', [1, 0, 0, 1]),
        for i, f in enumerate(mesh_files):
            # mesh_color = [0, 0, 1, 1]
            # mesh_color = np.random.rand(3).tolist() + [1.0]  # Random RGB color with alpha = 1.0
            geom = self.obj_body.add_geom(
                type = mujoco.mjtGeom.mjGEOM_MESH,
                meshname = f"{config.get('obj_name')}_mesh_{i}",
                condim = config.get('contact').get('condim', 3),
                quat = config.get('attach_pose')['quaternion'],
                rgba = mesh_color[0],
                density = material_list[config.get('material')]['density'],
                solref = material_list[config.get('material')]['solref'],
                friction = [self.friction, 0.005, 0.0001], # sliding friction between the two task objects
            )
            mesh = self._mj_spec.add_mesh()
            mesh.name = f"{config.get('obj_name')}_mesh_{i}"
            mesh.file = f
            mesh.scale = config.get('scale')

        print("loaded mesh files")


class MeshObject(BaseObject):
    
    def __init__(self,
                 mj_spec,
                 config_dict: dict,
                 friction: float):
        super(MeshObject, self).__init__(mj_spec, config_dict, friction)
        self.load_mesh_object(self._config)
        
    def load_mesh_object(self, config):

        # load the mesh files
        self.obj_body.add_geom(
            type = mujoco.mjtGeom.mjGEOM_MESH,
            meshname = f"{config.get('obj_name')}_mesh",
            condim = config.get('contact').get('condim', 3),
            rgba = config.get('mesh_color', [1, 0, 0, 1]),
            # mass = config.get('mass'),
            density = material_list[config.get('material')]['density'],
            solref = material_list[config.get('material')]['solref'],
            friction = [self.friction, 0.005, 0.0001],
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


class FlexcompObject(BaseObject):
    def __init__(self,
                mj_spec,
                config_dict: dict,
                friction: float):
        super(FlexcompObject, self).__init__(mj_spec, config_dict, friction)
        
        _mp = MeshObjects(obj_path=self._config.get('mesh_path'))
        _mp.convert_stl_to_msh()
        self.msh_path = _mp.output_msh_path

        self.load_flexcomp_object(self._config)
    
    def load_flexcomp_object(self, config):
        
        # compile spec
        self._mj_model = self._mj_spec.compile()

        # save to xml
        xmlstring = self._mj_spec.to_xml()
        root = ET.fromstring(xmlstring)

        # parse flexcomp to xml
        element_body = root.findall(".//*[@name='" + self.obj_body.name + "']")[0] # unique body name

        element_flexcomp = ET.SubElement(element_body, "flexcomp")
        element_flexcomp.set("rgba", " ".join([str(c) for c in config['mesh_color']]))
        element_flexcomp.set("scale", " ".join([str(c) for c in config['scale']]))
        element_flexcomp.set("radius", "0.00001")
        element_flexcomp.set("dim", "3")
        element_flexcomp.set("file", self.msh_path)
        element_flexcomp.set("mass", f"{config['mass']}") #TODO: set mass by material density and volume
        element_flexcomp.set("name", self.obj_body.name)
        element_flexcomp.set("type", "gmsh")

        element_contact = ET.SubElement(element_flexcomp, "contact")
        element_contact.set("condim", "1")
        element_contact.set("selfcollide", "none")
        element_contact.set("solimp", ".95 .99 .0001")
        element_contact.set("solref", "0.01 1")

        element_edge = ET.SubElement(element_flexcomp, "edge")
        element_edge.set("damping", "1")

        element_plugin = ET.SubElement(element_flexcomp, "plugin")
        element_plugin.set("plugin", "mujoco.elasticity.solid")

        element_config = ET.SubElement(element_plugin, "config")
        element_config.set("key", "young")   
        element_config.set("value", "200e7")

        element_config = ET.SubElement(element_plugin, "config")
        element_config.set("key", "poisson")   
        element_config.set("value", "0.2")         
        
        new_xmlstring = ET.tostring(root)

        # load spec from updated xml string
        self._mj_spec.from_string(new_xmlstring)
        
        print(f"loaded object {config.get('obj_name')}")


if __name__ == "__main__":
    mesh_stl_path = "/workspace/qbit/assets/task_env/primitives/box_5.013x20.853x5.204/box_5.013x20.853x5.204_male.stl"
    mesh_gmsh_path = mesh_stl_path[:-3] + "msh"


