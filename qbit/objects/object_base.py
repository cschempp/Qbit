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
from scipy.spatial.transform import Rotation as R

import mujoco
import mujoco.viewer

from qbit.utils.tf_utils import T
from qbit.utils.mujoco_utils import convert_quat_to_xyzw
from qbit.utils.mesh_processing import MeshObjects





class BaseObject:
    def __init__(self, 
                 mj_spec, 
                 config_dict: dict,
                 friction: float):
        
        self._mj_spec = mj_spec
        self._config = config_dict
        self.friction = friction

        # o_solref="0.02 1"
        self.material_list = {
            "steel": {
                "solref": [0.0012, 0.5], #-0.1
                "density": 7850,
                "young": 200e9,
                "poisson": 0.28,
            },
            "plastic": {
                "solref": [0.0018, 0.5], #-0.01
                "density": 7850,#1190,
                "young": 2.5e9,
                "poisson": 0.4,
            },
            "wood": {
                "solref": [0.0022, 0.5], #-0.01
                "density": 7850,#700,
                "young": 15e9,
                "poisson": 0.43,
            },
            "rubber": {
                "solref": [0.0028, 0.5],
                "density": 7850,#920,
                "young": 0.05e9,
                "poisson": 0.49,
            },
            "normal": {
                "solref": [0.02, 1.0],
                "density": 1000,
                "young": 5e5,
                "poisson": 0.25,
            },
        }

        self.start_position_hole, self.insertion_depth = self.get_hole_pose_depth(self._config)
        self.attach_body(config=self._config)

    def get_hole_pose_depth(self, config):
        self.mesh_path = config.get('mesh_path')
        
        meshfile = self.mesh_path

        mesh = trimesh.load_mesh(meshfile)
        # mesh = mesh.subdivide_loop(iterations=0)
        # mesh.export(self._config.get('mesh_path')[:-4]+"_processed.stl")

        mesh.vertices *= np.array(config.get('scale'))

        self._obj_volume = mesh.volume
        self._obj_mass = 0.1 #self._obj_volume * self.material_list[config["material"]]["density"]

        quat = config["attach_pose"]["quaternion"]
        quat = [quat[1], quat[2], quat[3], quat[0]]
        rotation_matrix = R.from_quat(quat).as_matrix()

        # Create a 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix

        # Apply the rotation
        mesh.apply_transform(transform)

        insertion_depth = mesh.extents[2]
        start_position_hole = np.array(config.get('attach_pose')['position']) + np.array([0, 0, insertion_depth/2 + 0.0000])
        
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
        _mp.decomposition_with_coacd(threshold=0.01)
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
                rgba = mesh_color[0],
                density = self.material_list[config.get('material')]['density'],
                solref = self.material_list[config.get('material')]['solref'],
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
            mass = self._obj_mass, #config.get('mass'),
            # density = self.material_list[config.get('material')]['density'],
            solref = self.material_list[config.get('material')]['solref'],
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
        # self.msh_path = self._config.get('mesh_path')[:-4]+"_processed.stl"

        self.nodes = self.parse_nodes_from_msh(file_path=self.msh_path)
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
        element_flexcomp.set("mass", str(self._obj_mass))
        element_flexcomp.set("name", self.obj_body.name)
        element_flexcomp.set("type", "gmsh")

        element_contact = ET.SubElement(element_flexcomp, "contact")
        element_contact.set("condim", "1")
        element_contact.set("selfcollide", "none") # bvh
        element_contact.set("internal", "false")
        # element_contact.set("activelayers", "1")
        element_contact.set("solimp", "0.95 0.99 0.001 0.5 2") # 0.0001
        element_contact.set("solref", " ".join([str(c) for c in self.material_list[config["material"]]["solref"]])) # "0.01 1"

        element_edge = ET.SubElement(element_flexcomp, "edge")
        element_edge.set("damping", "0.5")
        element_edge.set("equality", "true")
        # element_edge.set("solimp", "0.95 0.99 0.001 0.5 2") # 0.0001
        # element_edge.set("solref", " ".join([str(c) for c in self.material_list[config["material"]]["solref"]])) # "0.01 1"

        element_plugin = ET.SubElement(element_flexcomp, "plugin")
        element_plugin.set("plugin", "mujoco.elasticity.solid")

        element_config_0 = ET.SubElement(element_plugin, "config")
        element_config_0.set("key", "young")   
        element_config_0.set("value", str(self.material_list[config["material"]]["young"]))

        element_config_1 = ET.SubElement(element_plugin, "config")
        element_config_1.set("key", "poisson")   
        element_config_1.set("value", str(self.material_list[config["material"]]["poisson"]))

        # pin all the points which are at the bottom of the flexobject
        z_threshold = np.array(self.nodes)[:,3].min()
        pinned_node_indices = [i for i, x, y, z in self.nodes if z <= z_threshold]
        element_pin = ET.SubElement(element_flexcomp, "pin")
        element_pin.set("id", " ".join([str(c) for c in pinned_node_indices]))

        
        new_xmlstring = ET.tostring(root)

        # load spec from updated xml string
        self._mj_spec.from_string(new_xmlstring)
        
        print(f"loaded object {config.get('obj_name')}")

    def parse_nodes_from_msh(self, file_path):
        nodes = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            in_nodes_section = False
            for i, line in enumerate(lines):
                if "$Nodes" in line:
                    num_nodes = int(lines[i+1])
                    for j in range(num_nodes):
                        parts = lines[i+2+j].strip().split()
                        if len(parts) >= 4:
                            _, x, y, z = parts
                            nodes.append((j, float(x), float(y), float(z)))
                    break
        return nodes


class SpheredObject(BaseObject):
    """
    Todo: git clone SpheredDecomposition git project and use the class to load and decomp etc.
    right now only loading to test.

    """
    def __init__(self,
                 mj_spec,
                 config_dict, 
                 friction):
        super(SpheredObject, self).__init__(mj_spec, config_dict, friction)

        self._sphered_object_dir = os.path.join(*self._config.get('mesh_path').split(os.sep)[:-1], "sphered_decomposition", "cylinder_with_ring_0.04.npy")

        self.load_sphered_object(config=self._config)

    def load_sphered_object(self, config):
        decomposed_mesh = np.load( self._sphered_object_dir, allow_pickle=True)
        scale = np.array(config.get('scale'))

        self.FINAL_POINTS = decomposed_mesh.item()["points"] * scale
        self.FINAL_RADII = decomposed_mesh.item()["radii"] * scale[0]
        self.FINAL_COLORS = decomposed_mesh.item()["colors"]

        for point, radius, color in zip(self.FINAL_POINTS, self.FINAL_RADII, self.FINAL_COLORS):
            
            if color[2] > 200: material = "rubber"
            else: material = "steel"

            geom = self.obj_body.add_geom(
                type = mujoco.mjtGeom.mjGEOM_SPHERE,
                condim = config.get('contact').get('condim', 3),
                rgba = color.tolist() + [1.0],
                size = [radius]*3,
                pos = list(point),
                density = self.material_list[material]['density'],
                solref = self.material_list[material]['solref'],
                friction = [self.friction, 0.005, 0.0001], # sliding friction between the two task objects
            )

        print("loaded sphered object")

if __name__ == "__main__":
    mesh_stl_path = "/workspace/qbit/assets/task_env/primitives/box_5.013x20.853x5.204/box_5.013x20.853x5.204_male.stl"
    mesh_gmsh_path = mesh_stl_path[:-3] + "msh"


