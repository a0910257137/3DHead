from scipy.io import loadmat
from .bfm09l import BFM09


class ModelFactory:
    def __init__(self, config):
        self.config = config

    def build_model(self):
        recon_model = self.get_model()
        return recon_model

    def get_model(self):
        model_type = self.config.model_type
        model_path = self.config.model_path
        if model_type == 'bfm09':
            model_dict = loadmat(model_path)
            recon_model = BFM09(self.config, model_dict)
        return recon_model
