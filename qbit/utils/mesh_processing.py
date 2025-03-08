"""
    Utils for mesh processing    
    - CoACD: https://github.com/SarahWeiii/CoACD
"""
import os
import glob
import numpy as np
import trimesh
import coacd


class MeshObjects:
    
    
    def __init__(self, obj_path):
        
        self._obj_path = obj_path
        self._decomposed_mesh_dir_name = "decomposed_fine"
        
        self._decomposed_mesh_dir = os.path.join(
            os.path.dirname(os.path.abspath(self._obj_path)),
                            self._decomposed_mesh_dir_name)

    def show_decomposed_mesh(self):

        mesh_files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(self._obj_path)), 
                                            self._decomposed_mesh_dir_name, "*.obj"))        
        scene = trimesh.Scene()
        for i, f in enumerate(mesh_files):
            
            mesh = trimesh.load(f, process=False)
                        
            if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
                mesh.fix_normals()
            mesh.visual.material = trimesh.visual.material.SimpleMaterial(
                diffuse=[1.0, 0.0, 0.0, 1.0]  # Red diffuse color (values in 0-1)
            )                
            mesh.visual = trimesh.visual.ColorVisuals(mesh)
                
            mesh.visual.vertex_colors = np.tile([255, 0, 0, 255], (mesh.vertices.shape[0], 1))
            mesh.visual.face_colors[:, :4] = (np.random.rand(4) * 255).astype(np.uint8)
            
            scene.add_geometry(mesh, node_name=f"part_{i}")

        scene.show(flags={'wireframe':True})


    def show_mesh(self):
        scene = trimesh.Scene()
        mesh = trimesh.load(self._obj_path)
        mesh.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
        scene.add_geometry(mesh)
        
        scene.show(flags={'wireframe':True})


    def decomposition_with_vhacd(self):
        mesh = trimesh.load(self._obj_path)
        decomposed_meshes = trimesh.decomposition.convex_decomposition(mesh)
        
        scene = trimesh.Scene()
        for i, p in enumerate(decomposed_meshes):
            m = trimesh.Trimesh(p['vertices'], p['faces'])
            m.visual.face_colors = (np.random.rand(3) * 255).astype(np.uint8)
            m.visual.vertex_colors = (np.random.rand(3) * 255).astype(np.uint8)
            m.export(f"/workspace/qbit/assets/task_env/plugs/vhacd_usb_a_female/part_{i}.obj")
            scene.add_geometry(m)
        # scene.show(flags={'wireframe':True})
        # print("Decomposed meshes: ", decomposed_meshes)


    def decomposition_with_coacd(self, threshold=0.02):
        
        if not os.path.exists(self._decomposed_mesh_dir):
            os.makedirs(self._decomposed_mesh_dir)
            
        mesh = trimesh.load(self._obj_path)
        mesh = coacd.Mesh(mesh.vertices, mesh.faces)
        parts = coacd.run_coacd(mesh, threshold=threshold)
        
        for i, p in enumerate(parts):
            mesh = trimesh.Trimesh(p[0], p[1])
            mesh.visual.face_colors = trimesh.visual.color.random_color()
            mesh.export(f"{self._decomposed_mesh_dir}/part_{i}.obj")
        print(f"Decomposed meshes ({i} parts) are saved in {self._decomposed_mesh_dir}")
        

if __name__ == "__main__":
    

    obj_path = "/workspace/qbit/assets/task_env/plugs/usb_a_female_m.stl"
    # obj_path = "/workspace/qacbi/assets/task_env/plugs/dsub_25_female.stl"
    
    mo = MeshObjects(obj_path)
    mo.decomposition_with_vhacd()
    