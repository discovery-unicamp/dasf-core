import torch

from .base import BaseLoader


class TorchLoader(BaseLoader):
    def __init__(self, model_class, dtype=torch.float32, checkpoint=None, device=None):
        super().__init__()
        self.model_class = model_class
        self.dtype = dtype
        self.checkpoint = checkpoint
        self.device = device

    def load_model(self, **kwargs):
        model = self.model_class(**kwargs)
        device = torch.device("cuda" if self.device == "gpu" else "cpu")
        if self.checkpoint:
            state_dict = torch.load(self.checkpoint, map_location=device)
            state_dict = (
                state_dict["state_dict"] if "state_dict" in state_dict else state_dict
            )  # In case model was saved by TensorBoard
            model.load_state_dict(state_dict)
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
