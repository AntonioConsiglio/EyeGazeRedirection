import torch
import os
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import writer
from model.dataset import GazeDataset
from model.gaze_direction import GazeCorrection
from datetime import datetime
import numpy as np
import cv2

dataset = GazeDataset(r"C:\Users\anton\Desktop\PROGETTI\EyeGazeRedirection\MPIIGaze\training",
                      mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5))
dataloader = DataLoader(dataset=dataset,
                        batch_size=1,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=False,
                        drop_last=False)
    
model = GazeCorrection()

weights = torch.load("./last_weights_62.pth",map_location="cpu")
model.load_state_dict(weights)
del model.discriminator
del model.vgg16
model = model.cuda()
model.eval()

def denormilize(image):
    std = torch.tensor((0.5, 0.5, 0.5)).unsqueeze(1).unsqueeze(2)
    mean = torch.tensor((0.5, 0.5, 0.5)).unsqueeze(1).unsqueeze(2)
    return image.cpu()*std + mean

with torch.no_grad():
    for it, (real_images,r_angles,_,target_images,t_angles) in enumerate(dataloader,start=1):
        # Convert data to PyTorch tensors
        real_images = real_images.to(model.device)

        maxv = torch.max(real_images)
        minv = torch.min(real_images)

        r_angles = r_angles.to(model.device)
        target_images = target_images.to("cpu")
        t_angles = t_angles.to(model.device)
        t_angles_my = torch.zeros_like(t_angles)
        t_angles_my[0][0] = -35/35
        t_angles_my[0][1] = 35/35

        image = model.forward(real_images,t_angles_my)

        print(t_angles_my*35)

        rtos = denormilize(real_images.squeeze()).cpu().numpy()*255
        rtos = rtos.transpose(1,2,0).astype(np.uint8)

        predimg = (denormilize(image.squeeze()).cpu().numpy()*255).transpose(1,2,0).astype(np.uint8)
        # predimg = ((image.squeeze()).cpu().numpy()*255).transpose(1,2,0).astype(np.uint8)

        target_image = (denormilize(target_images.squeeze()).cpu().numpy()*255).transpose(1,2,0).astype(np.uint8)

        merged = cv2.cvtColor(cv2.resize(np.vstack([
                                        np.hstack([rtos,np.zeros_like(rtos)]),
                                        np.hstack([predimg,target_image])
                                        ]),
                                        (0,0),fx=2,fy=2),cv2.COLOR_RGB2BGR)
        cv2.imshow("init",merged)
        cv2.waitKey(0)