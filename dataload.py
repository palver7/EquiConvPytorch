import pandas as pd
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

imgs = os.listdir('train/RGB')
CMimgs = os.listdir("train/CM_gt")
EMimgs = os.listdir("train/EM_gt")
imgpaths = []
CMpaths = []
EMpaths = []
root=""
#abspth = os.path.abspath(root)
for img in imgs:
    pths = os.path.join(root,"train/RGB/",img)
    imgpaths.append(pths)
for CMimg in CMimgs:
    pths = os.path.join(root,"train/CM_gt/",CMimg)
    CMpaths.append(pths)
for EMimg in EMimgs:
    pths = os.path.join(root,"train/EM_gt/",EMimg)
    EMpaths.append(pths)    
dict={'images' : imgpaths, 'EM' : EMpaths, 'CM' : CMpaths}
df = pd.DataFrame(data = dict)
#df.to_json("imagedata.json")
image = Image.open(df['images'][0])
image = np.asarray(image)
#print(image.shape)
EM = Image.open(df['EM'][0])
CM = Image.open(df['CM'][0])
EM = np.asarray(EM)
EM = np.expand_dims(EM,axis=2)
CM = np.asarray(CM)
CM = np.expand_dims(CM,axis=2)
ECM = np.concatenate((EM,CM),axis = 2)
#print (ECM.shape)
maps = Image.fromarray(ECM)