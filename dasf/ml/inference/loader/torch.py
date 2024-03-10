import inspect
import os

import torch

from .base import BaseLoader


class TorchLoader(BaseLoader):
    """
    Model Loader for Torch models
    """

    def __init__(
        self, model_class_or_file, dtype=torch.float32, checkpoint=None, device=None
    ):
        """
        model_class_or_file: class or file with model definition
        dtype: data type of model input
        checkpoint: model chekpoint file
        device: device to place model ("cpu" or "gpu")
        """
        super().__init__()
        self.model_class_or_file = model_class_or_file
        self.dtype = dtype
        self.checkpoint = checkpoint
        self.device = device

    def load_model(self, **kwargs):
        device = torch.device("cuda" if self.device == "gpu" else "cpu")
        if inspect.isclass(self.model_class_or_file):
            model = self.model_class_or_file(**kwargs)
            if self.checkpoint:
                state_dict = torch.load(self.checkpoint, map_location=device)
                state_dict = (
                    state_dict["state_dict"]
                    if "state_dict" in state_dict
                    else state_dict
                )  # In case model was saved by TensorBoard
                model.load_state_dict(state_dict)
        elif os.path.isfile(self.model_class_or_file):
            model = torch.load(self.model_class_or_file)
        else:
            raise ValueError(
                "model_class_or_file must be a model class or path to model file"
            )
        model.to(device=device)
        model.eval()
        return model

    def inference(self, model, data):
        data = torch.from_numpy(data)
        device = torch.device("cuda" if self.device == "gpu" else "cpu")
        data = data.to(device, dtype=self.dtype)
        with torch.no_grad():
            output = model(data)
        return output.cpu().numpy() if self.device == "gpu" else output.numpy()
