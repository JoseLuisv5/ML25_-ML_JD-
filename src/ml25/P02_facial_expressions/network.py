import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib
from torchvision.models import resnet18, ResNet18_Weights

file_path = pathlib.Path(__file__).parent.absolute()


def build_backbone(model="resnet18", weights="imagenet", freeze=True, last_n_layers=2):
    if model == "resnet18":
        backbone = resnet18(pretrained=weights == "imagenet")
        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False
        return backbone
    else:
        raise Exception(f"Model {model} not supported")


class Network(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # TODO: Calcular dimension de salida
        # ---- 1) Calcular la dimensión de salida (ancho/alto) después de conv+pool ----
        d = input_dim  # empieza en 48

        # Conv1: kernel=3, padding=1, stride=1 -> mantiene tamaño
        d = self.calc_out_dim(d, kernel_size=3, stride=1, padding=1)  # 48
        # Pool1: kernel=2, stride=2, sin padding -> divide entre 2
        d = self.calc_out_dim(d, kernel_size=2, stride=2, padding=0)  # 24

        # Conv2: igual que conv1, mantiene tamaño
        d = self.calc_out_dim(d, kernel_size=3, stride=1, padding=1)  # 24
        # Pool2: otra vez a la mitad
        d = self.calc_out_dim(d, kernel_size=2, stride=2, padding=0)  # 12

        out_dim = 64 * d * d  # 64 canales, 12x12 -> 64*12*12 = 9216

        # ---- 2) Definir las capas de la red ----

        # Capas convolucionales
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Capas fully-connected
        self.fc1 = nn.Linear(out_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_classes)

        self.to(self.device)


    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = math.floor((in_dim - kernel_size + 2 * padding) / stride) + 1
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Define la propagacion hacia adelante de tu red
        # Asegurar que esté en el device correcto
        x = x.to(self.device)

        # Si viene sin dimensión de batch (C,H,W) lo convertimos a (1,C,H,W)
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Bloque conv1 + pool
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        # Bloque conv2 + pool
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Aplanar a (batch, features)
        x = torch.flatten(x, start_dim=1)

        # Fully-connected
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        proba = F.softmax(logits, dim=1)

        return logits, proba

    def predict(self, x):
        with torch.inference_mode():
            return self.forward(x)

    def save_model(self, model_name: str):
        """
        Guarda el modelo en el path especificado
        args:
        - net: definición de la red neuronal (con nn.Sequential o la clase anteriormente definida)
        - path (str): path relativo donde se guardará el modelo
        """
        models_path = file_path / "models" / model_name
        if not models_path.parent.exists():
            models_path.parent.mkdir(parents=True, exist_ok=True)
        # TODO: Guarda los pesos de tu red neuronal en el path especificado
        torch.save(self.state_dict(), models_path)

    def load_model(self, model_name: str):
        """
        Carga el modelo en el path especificado
        args:
        - path (str): path relativo donde se guardó el modelo
        """
        # TODO: Carga los pesos de tu red neuronal
        models_path = file_path / "models" / model_name
        state_dict = torch.load(models_path, map_location=self.device)
        self.load_state_dict(state_dict)
        self.to(self.device)
        self.eval()  # deja el modelo en modo evaluación
