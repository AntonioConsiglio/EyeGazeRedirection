import os
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import torch
from os.path import join,isdir
import numpy as np
from tqdm import tqdm

class GazeDataset(Dataset):
    def __init__(self, data_path,mean=None,std=None, ids=15,transform=None):
        self.root = data_path
        self.images_root = join(self.root,"images")
        self.labels_root = join(self.root,"labels")
        self.ids = ids
        if transform is not None:
            self.transform = transform
        else:
            if mean is None or std is None:
                self.mean, self.std = self.calculate_mean_and_std(data_path)
            else:
                self.mean = mean
                self.std = std
                

            self.transform = transforms.Compose([
            # transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
            ])
            self.tensor_mean = torch.tensor(list(self.mean)).unsqueeze(1).unsqueeze(2)
            self.tensor_std = torch.tensor(list(self.std)).unsqueeze(1).unsqueeze(2)

        self._create_datas()
        

    def _create_datas(self):
        self.images_path = []
        self.labels_path = []
        self.ids_lbl = []
        for n,path in enumerate(os.listdir(self.images_root),start=1):
            for j in os.listdir(join(self.images_root,path)):
                if j.endswith(".jpg"):
                    self.images_path.append(join(self.images_root,path,j))
                    self.labels_path.append(join(self.labels_root,path,j.replace("jpg","txt"))) 
                    self.ids_lbl.append(n)
        self.ids_lbl = np.array(self.ids_lbl)

        self._create_angles_tensors()

    def _create_angles_tensors(self):
        self.labels_dict = {}
        #normal considering max angle 35°
        for n,lbl_path in enumerate(self.labels_path):
            with open(lbl_path,"r") as f:
                data = f.readline().replace("\n","").split(" ")
            
            angles = torch.tensor([float(angle)/35 for angle in data])
            self.labels_dict[n] = angles
            
    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):

        r_image = self._to_image(self.images_path[index])
        lbl = self.ids_lbl[index]
        r_angles = self.labels_dict[index]

        t_idx = self._get_t_idx(lbl)
        t_image = self._to_image(self.images_path[t_idx])
        t_angles = self.labels_dict[t_idx]
        
        return r_image,r_angles,lbl,t_image,t_angles
    
    def _get_t_idx(self,person):
        new_idx = np.where(self.ids_lbl == person)[0]
        np.random.shuffle(new_idx)
        
        return new_idx[np.random.randint(0,len(new_idx))]

    def _to_image(self, file_name):
        img = Image.open(file_name).convert("RGB")
        
        img = self.transform(img)

        return img

    def denormalize_color(self,image):
        return image*self.tensor_std + self.tensor_mean

    @classmethod
    def calculate_mean_and_std(cls,data_path):
        transform = transforms.Compose([
            transforms.ToTensor(),]
        )

        print("#"*40," No mean and std detected! Start the calculation now! ","#"*40)
        dataset = cls(data_path=data_path,mean=(0,0,0),std=(1,1,1),transform=transform)
        dataloader = DataLoader(dataset=dataset,batch_size=32,shuffle=False,num_workers=0)
        loop = tqdm(dataloader)
        loop.set_description(f"Calculating Mean and Std for dataset -- {data_path} -- ")

        cnt = 0
        mean = torch.empty(3)
        sum_of_squares_n = torch.empty(3)

        for images,_,_,_,_ in loop:
            b, c, h, w = images.shape
            nb_pixels = b*h*w
            sum_ = torch.sum(images, dim=[0, 2, 3])
            sum_of_square = torch.sum(images ** 2,
                                    dim=[0, 2, 3])
            mean = (cnt * mean + sum_) / (
                        cnt + nb_pixels)
            sum_of_squares_n = (cnt * sum_of_squares_n + sum_of_square) / (
                                cnt + nb_pixels)
            cnt += nb_pixels

        
        std = tuple(torch.sqrt(sum_of_squares_n - mean ** 2).numpy().tolist())
        mean = tuple(mean.numpy().tolist())  

        print("\n Mean and Std for dataset -- {} -- is equal to: \n MEAN: {} ,  STD: {}".format(data_path,mean,std))

        return mean,std

if __name__ == "__main__":

    dataset = GazeDataset(r"C:\Users\anton\Desktop\PROGETTI\EyeGazeRedirection\MPIIGaze\training")



    