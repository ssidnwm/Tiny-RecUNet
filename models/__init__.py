from .UNet import UNet
from .SwinUnet import SwinUnet
from .TransUnet import TransUnet
from .rec_unet import TinyRecUNet

model_dict = {
    "UNet": UNet,
    "SwinUnet": SwinUnet,
    "TransUnet": TransUnet,
    "tinyrecunet": TinyRecUNet,
}
