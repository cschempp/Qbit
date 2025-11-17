import numpy as np
import os
import matplotlib.pyplot as plt
from qbit.sim_envs.mujoco_env_base import MujocoEnvBase
from datetime import datetime


class DataRecording():
    def __init__(self, task_env_config_path):
        self.init()

        self.config = MujocoEnvBase.parse_qbit_config_yaml(task_env_config_path)

        self.RESULT_DIR = os.path.join("/workspace/examples/experiment_results/", self.config["data_recording"]["save_folder"])

        if not os.path.exists(self.RESULT_DIR):
            os.makedirs(self.RESULT_DIR)

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

    def save(self,):
        self.filename = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.savepath = os.path.join(self.RESULT_DIR, self.filename)

        self.timestamp = np.array(self.timestamp)
        self.eef_fts = np.array(self.eef_fts)
        self.eef_pos = np.array(self.eef_pos)
        self.eef_qua = np.array(self.eef_qua)
        self.joint_states = np.array(self.joint_states)

        np.savez(self.savepath + ".npz",
                 timestamp = self.timestamp,
                 eef_fts = self.eef_fts,
                 eef_pos = self.eef_pos,
                 eef_qua = self.eef_qua,
                 joint_states = self.joint_states)
    
        self.print_info()
    
    def print_info(self):
        print("recorded data saved to " + self.savepath)

        print("timestamp: " + str(self.timestamp.shape))
        print("eef_fts: " + str(self.eef_fts.shape))
        print("eef_pos: " + str(self.eef_pos.shape))
        print("eef_qua: " + str(self.eef_qua.shape))
        print("joint_states: " + str(self.joint_states.shape))

    def plot_data(self):
        # load previously saved .npz file
        data = np.load(self.savepath + ".npz")

        # map saved arrays to names used by the plotting code
        time = np.asarray(data["timestamp"])
        force = np.asarray(data["eef_fts"])
        position = np.asarray(data["eef_pos"])

        # ensure arrays are 2D where expected
        if force.ndim == 1:
            force = force.reshape(-1, 1)
        if position.ndim == 1:
            position = position.reshape(-1, 1)

        labels = ["Fx", "Fy", "Fz", "x", "y", "z"]

        for i,j in enumerate((1, 2, 3)):
            plt.subplot(2,3,j)
            plt.plot(time, force[:,i]-force[0,i], label=labels[j-1])
            plt.legend(loc="upper right")

        for i,j in enumerate((4, 5, 6)):
            plt.subplot(2,3,j)
            plt.plot(time, position[:,i]-position[0,i], label=labels[j-1])
            plt.legend(loc="upper right")
        
        plt.tight_layout()
        plt.savefig(self.savepath + ".png")
        plt.close()
        plt.cla()
        plt.clf()

        return

   
