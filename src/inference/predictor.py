import torch
from PIL import Image

from src.models.cnn import CNN
from src.data.preprocess import get_transform
from config.config import load_config


class Predictor:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        config = load_config()

        self.threshold = config["model"]["threshold"]

        self.transform = get_transform()

        self.model = self.load_model()


    def load_model(self):

        model = CNN().to(self.device)

        model.load_state_dict(
            torch.load("outputs/models/best_model.pth", map_location=self.device)
        )

        model.eval()

        return model


    def preprocess(self, image_path):

        image = Image.open(image_path)

        x = self.transform(image)

        x = x.unsqueeze(0)

        x = x.to(self.device)

        return x


    def predict(self, image_path):

        x = self.preprocess(image_path)

        with torch.no_grad():

            prob = self.model(x)

            prob = prob.item()

            pred = 1 if prob >= self.threshold else 0

        return prob, pred