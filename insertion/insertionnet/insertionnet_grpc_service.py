"""
Provide the insertionnet as a gRPC service for large-scale testing.
"""

import keras

from qbit.interfaces.grpc.task_service_proxy import QbitTaskService


class InsertionNetService(QbitTaskService):
    """
    Inherit from the base task service 
    and change the inference callback to load the insertionnet model.
    """
    
    def __init__(self,
                 model_path,
                 insecure_port=50052,
                 max_workers=4):
        
        self._model_path = model_path
        
        super().__init__(insecure_port, max_workers)

        print("Starting InsertionNetService")


    def inference_cb(self):
        """
        Load the model and return the callback function for inference.
        """
        insertion_net = keras.saving.load_model(
            self._model_path,
            custom_objects=None,
            compile=True,
            safe_mode=True)

        return insertion_net    
        

if __name__ == "__main__":
    
    model_path = "/workspace/insertion/insertionnet/checkpoints/insertionnet_IF_sim_peg"
    
    service = InsertionNetService(
        model_path=model_path,
    )
    
    service.start()
