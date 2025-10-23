from .UNet import UNet
from .SwinUnet import SwinUnet
from .TransUnet import TransUnet

model_dict = {
    "UNet": UNet,
    "SwinUnet": SwinUnet,
    "TransUnet": TransUnet,
}
