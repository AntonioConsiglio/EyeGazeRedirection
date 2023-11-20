import torch
import os
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from model.dataset import GazeDataset
from model.gaze_direction import GazeCorrection
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR

BATCH_SIZE = 16
BETA1 = 0.9
BETA2 = 0.999
LR = 5e-04
EPOCHS = 300
CHECKPOINT = None #
# f = (1e-8/2e-4)**(1/(300-150))

def lr_lambda(epoch):

    if epoch < 150:
        return 1.0
    else:
        return 0.93611
    

if __name__ == "__main__":
    
    today = datetime.today().strftime("%Y%m%d_%H%M%S")
    logsf = os.path.join("logs",today)

    os.makedirs(logsf)
    os.makedirs(os.path.join(logsf,"weights"))
    os.makedirs(os.path.join(logsf,"checkpoints"))
    
    writer =  SummaryWriter(log_dir=logsf)
    
    print("#"*40," LOADING DATASET ","#"*40)
    dataset = GazeDataset(r"C:\Users\anton\Desktop\PROGETTI\EyeGazeRedirection\MPIIGaze\training",
                         mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5))
    
    dataloader = DataLoader(dataset=dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=True)
    
    print("#"*40," LOADING MODEL ","#"*40)
    model = GazeCorrection()

    optimizers = {
        "d":Adam(model.discriminator.parameters(),lr=LR,betas=(BETA1,BETA2)),
        "g":Adam(model.generator.parameters(),lr=LR,betas=(BETA1,BETA2))
    }

    schedulers = {
        "d":LambdaLR(optimizer=optimizers["d"],lr_lambda=lr_lambda),
        "g":LambdaLR(optimizer=optimizers["g"],lr_lambda=lr_lambda),
    }
    print("#"*40," START TRAINING ","#"*40)
    model.train_(dataloader,optimizers=optimizers,restore_chkpt=CHECKPOINT,schedulers=schedulers
                 ,summary_writer=writer,logspath=logsf,epochs=EPOCHS)