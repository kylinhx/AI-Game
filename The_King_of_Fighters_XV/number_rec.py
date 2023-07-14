import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import numpy as np
import cv2 as cv


class NumberRec(nn.Module):
    def __init__(self):
        super(NumberRec,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,16,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 2, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(128, 10) #10分类的问题
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
 
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.contiguous().view(x.shape[0], -1)
        x = self.out(x)
        return x

    def load_weights(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path)
        except:
            raise(Exception("ERROR:checkpoint does't exist!"))
        self.load_state_dict(checkpoint)
        self.to(self.device)

    def predict(self, src:list) -> np.ndarray:
        temp = []
        for i in range (len(src)):
            img_resize = cv.resize(src[i], (24, 32), fx=None, fy=None)
            temp.append(img_resize)
        data_4dim = np.array(temp)  
        data_4dim = np.transpose(data_4dim, (0, 3, 1, 2)) # bchw
        data_4dim_tensor = torch.from_numpy(data_4dim).float().to(self.device)
        output = self.forward(data_4dim_tensor)
        result = torch.max(output.cpu().detach(), dim=1)[1].numpy()

        return result
