"""
Test the insertionnet service using grpc.
"""

import time
import numpy as np
import grpc

from qbit.interfaces.grpc import qbit_pb2
from qbit.interfaces.grpc import qbit_pb2_grpc

import cv2


TEST_DATA_PATH = "/workspace/insertion/insertionnet/test_data/test_data_0.npy"

SERVICE_HOST = 'localhost:50052'


def run():
    
    data = np.load(TEST_DATA_PATH, allow_pickle=True).item()
    
    img = data['image']
    img = img[:, 500:]
    img = cv2.resize(img, dsize=(256, 256))
    img_np = np.array(img)
    
    img_bytes = img_np.tobytes()
    wrench = data['wrench']
    
    try:
        with grpc.insecure_channel(SERVICE_HOST) as channel:
            stub = qbit_pb2_grpc.QbitInterfaceStub(channel)

            response = stub.CheckServerConnection(qbit_pb2.Ping(ping=True))
            print("Connection to service provider: ", response.pong)

            start = time.time()
            response = stub.RequestNextActionForInsertion(qbit_pb2.ObsState(
                image_data=img_bytes,
                wrench=wrench,
                ))

            print(f"Action: {response.eef_action}")
            print(f"Inference time [Client]: {time.time() - start}")
            
    except Exception as e:
        print("Error occurred:", e)


if __name__ == '__main__':
    run()
