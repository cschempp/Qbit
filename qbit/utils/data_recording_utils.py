import numpy as np



class DataRecording():
    def __init__(self, object_type: str, material_male: str, material_female: str):
        self.init()

        # for PIPE
        self.object_type = object_type
        self.material_male = material_male
        self.material_female = material_female
        # self.trajectory_type = ""
        # self.trajectory_depth = ""
        # self.trajectory_duration = ""

    def init(self):
        self.timestamp = []
        self.eef_fts = []
        self.eef_pos = []
        self.eef_qua = []
        self.joint_states = []
    
    def record(self, timestamp, eef_fts, eef_pos, eef_qua, joint_states):
        self.timestamp.append(timestamp)
        self.eef_fts.append(eef_fts)
        self.eef_pos.append(eef_pos)
        self.eef_qua.append(eef_qua)
        self.joint_states.append(joint_states)

    def save(self, savepath = "data_recorded.npz"):
        self.savepath = savepath

        self.timestamp = np.array(self.timestamp)
        self.eef_fts = np.array(self.eef_fts)
        self.eef_pos = np.array(self.eef_pos)
        self.eef_qua = np.array(self.eef_qua)
        self.joint_states = np.array(self.joint_states)

        np.savez(self.savepath, 
                 timestamp = self.timestamp,
                 eef_fts = self.eef_fts,
                 eef_pos = self.eef_pos,
                 eef_qua = self.eef_qua,
                 joint_states = self.joint_states,
                 object_type = self.object_type,
                 material_male = self.material_male,
                 material_female = self.material_female)
        
        self.print_info()
    
    def print_info(self):
        print("recorded data saved to " + self.savepath)

        print("timestamp: " + str(self.timestamp.shape))
        print("eef_fts: " + str(self.eef_fts.shape))
        print("eef_pos: " + str(self.eef_pos.shape))
        print("eef_qua: " + str(self.eef_qua.shape))
        print("joint_states: " + str(self.joint_states.shape))

        print("object_type: " + self.object_type)
        print("material_male: " + self.material_male)
        print("material_female: " + self.material_female)
