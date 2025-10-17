import datetime

model_name = "unet"
exp_name = model_name + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

batch_size = 16
epochs = 50
lr = 0.0001
workers = 0
weights = "./"
image_size = 224
aug_scale = 0.05
aug_angle = 15
