import datetime

model_name = "TransUnet"
exp_name = model_name + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

batch_size = 16
epochs = 100
lr = 0.0001
workers = 4
weights = "./"
image_size = 224
aug_scale = 0.05
aug_angle = 15

model_args = {
    "UNet": {
        "in_channels": 3,
        "out_channels": 1,

    },
    "TransUnet": {
        "in_channels": 3,
        "out_channels": 1,
        "img_size": 224,
        "backbone": "resnet34",
        "pretrained_backbone": False,
        "embed_dim": 256,
        "depth": 6,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "patch_grid": None,          # e.g., (14,14) for 224x224; None = full grid (H/16, W/16)
        "embed_dropout": 0.1,        # e.g., 0.1 for embedding dropout like the paper
        "decoder_channels": (256, 128, 64, 16),
        "head_channels": 512,
    },
}