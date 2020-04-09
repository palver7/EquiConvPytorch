from PIL import Image
from skimage.feature import corner_peaks, peak_local_max
import numpy as np
import os
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F


def corners_2_xy(img):
    
    #img = img.copy()
    #img[img < 127] = 0
    #img[img > 127] = 255
    
    local_peaks = corner_peaks(img, min_distance=5, threshold_rel=0.5, indices=True)


    local_peaks = np.array(local_peaks, dtype=np.float64)
    height, width = img.shape
    width /=3
    col1m = (local_peaks[:,1]>=width) & (local_peaks[:,1]<2*width)
    peaks = local_peaks[col1m] 
    peaks[:,0]/=height
    peaks[:,1]-= width
    peaks[:,1]/= width
    return peaks
transf = transforms.Compose([transforms.Resize((112,112)),transforms.ToTensor()])
root = 'test/CM_gt/'
images = os.listdir(root)
for img in images:

    img_path = os.path.join(root,img)
    image = Image.open(img_path)
    
    tensor = transf(image)
    tensor1 = tensor * 255
    tensor1[tensor1<127] = 0
    tensor1[tensor1>127] = 255
    #tensor1=torch.unsqueeze(tensor1,dim=0)
    #kernel_tensor=torch.tensor([[[[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]]]])
    #tensor1=F.conv2d(tensor1,kernel_tensor,padding=(1,1))
    tensor1=torch.cat((tensor1,tensor1,tensor1),dim=-1)
    tensor1 = torch.squeeze(tensor1)
    imgarray = tensor1.numpy().astype(np.uint8)
    detection = corners_2_xy(imgarray)
    print(len(detection))
    #if len(detection) > 8:
    #    print(detection)
