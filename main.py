import os
import sys
import random
import datetime
import matplotlib.pyplot as plt
from skimage.io import imsave
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from models.unet import UNet
from dataset import (
    BrainSegmentationDataset, 
    transforms, crop_sample, pad_sample, resize_sample, normalize_volume, 
    Compose, data_loaders
    )
from utils import (
    Tee,
    DiceLoss,
    dsc_per_volume, dsc_distribution,
    postprocess_per_volume,
    log_scalar_summary, log_loss_summary,
    plot_dsc,
    gray2rgb,
    outline,
)
from config import *
from config import model_name, batch_size, epochs
from models import model_dict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # 중복 라이브러리 로드 방지


ModelClass = model_dict[model_name]
model = ModelClass(in_channels=BrainSegmentationDataset.in_channels, out_channels=BrainSegmentationDataset.out_channels)

def train_validate():
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
    print("using device:", device) 

    loader_train, loader_valid = data_loaders(batch_size, workers, image_size, aug_scale, aug_angle)
    loaders = {"train": loader_train, "valid": loader_valid}

    model.to(device)

    dsc_loss = DiceLoss()
    best_validation_dsc = 0.0

    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_train = []
    loss_valid = []
    
    step = 0
    
    train_loss_history = []
    valid_loss_history = []

    for epoch in range(epochs):
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            validation_pred = []
            validation_true = []

            for i, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1

                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)
                # if device.type == "cuda":
                #     print("GPU memory allocated:", torch.cuda.memory_allocated() // (1024*1024), "MB")

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = model(x)

                    loss = dsc_loss(y_pred, y_true)

                    if phase == "valid":
                        loss_valid.append(loss.item())
                        y_pred_np = y_pred.detach().cpu().numpy()
                        validation_pred.extend(
                            [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                        )
                        y_true_np = y_true.detach().cpu().numpy()
                        validation_true.extend(
                            [y_true_np[s] for s in range(y_true_np.shape[0])]
                        )
                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()

            if phase == "train":
                log_loss_summary(loss_train, epoch)
                train_loss_history.append(np.mean(loss_train))
                loss_train = []

            if phase == "valid":
                log_loss_summary(loss_valid, epoch, prefix="val_")
                valid_loss_history.append(np.mean(loss_valid))
                mean_dsc = np.mean(
                    dsc_per_volume(
                        validation_pred,
                        validation_true,
                        loader_valid.dataset.patient_slice_index,
                    )
                )
                log_scalar_summary("val_dsc", mean_dsc, epoch)
                if mean_dsc > best_validation_dsc:
                    best_validation_dsc = mean_dsc
                    torch.save(model.state_dict(), os.path.join(weights, f"{model_name}.pt"))
                loss_valid = []

    # Save loss curves after training
    plt.figure()
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(valid_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curve_{}.png'.format(exp_name))
    plt.close()
    
    print("\nBest validation mean DSC: {:4f}\n".format(best_validation_dsc))
    
    state_dict = torch.load(os.path.join(weights, f"{model_name}.pt"))
    model.load_state_dict(state_dict)
    model.eval()

    input_list = []
    pred_list = []
    true_list = []
    
    for i, data in enumerate(loader_valid):
        x, y_true = data
        x, y_true = x.to(device), y_true.to(device)
        with torch.set_grad_enabled(False):
            y_pred = model(x)
            y_pred_np = y_pred.detach().cpu().numpy()
            pred_list.extend([y_pred_np[s] for s in range(y_pred_np.shape[0])])
            y_true_np = y_true.detach().cpu().numpy()
            true_list.extend([y_true_np[s] for s in range(y_true_np.shape[0])])
            x_np = x.detach().cpu().numpy()
            input_list.extend([x_np[s] for s in range(x_np.shape[0])])
            
    volumes = postprocess_per_volume(
        input_list,
        pred_list,
        true_list,
        loader_valid.dataset.patient_slice_index,
        loader_valid.dataset.patients,
    )
    
    dsc_dist = dsc_distribution(volumes)

    dsc_dist_plot = plot_dsc(dsc_dist)
    imsave("./dsc.png", dsc_dist_plot)

    print("volumes keys:", list(volumes.keys()))
    for p in volumes:
        # p = "kaggle_3m/TCGA_DU_7014_19860618"
        # x shape: (28, 3, 224, 224)
            # 28: 한 환자(volume)에 포함된 슬라이스(slice) 개수 (즉, 3D 볼륨의 두께 방향)
        x = volumes[p][0]
        y_pred = volumes[p][1]
        y_true = volumes[p][2]
        for s in range(x.shape[0]):
            image = gray2rgb(x[s, 1])  # channel 1 is for FLAIR
            image = outline(image, y_pred[s, 0], color=[255, 0, 0])
            image = outline(image, y_true[s, 0], color=[0, 255, 0])
            p_id = p.split("/")[-1].split("\\")[-1]
            filename = "{}-{}.png".format(p_id, str(s).zfill(2))
            filepath = os.path.join("./result_img", filename)
            imsave(filepath, image)


if __name__ == "__main__":
    log_filename = f"{exp_name}.log"
    log_file = open(log_filename, 'w')
    sys.stdout = Tee(sys.__stdout__, log_file)
    train_validate()