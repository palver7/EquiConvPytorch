import pandas as pd
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

imgs = os.listdir('trainsmall/RGB')
CMimgs = os.listdir("trainsmall/CM_gt")
EMimgs = os.listdir("trainsmall/EM_gt")
imgpaths = []
CMpaths = []
EMpaths = []
root=""
#abspth = os.path.abspath(root)
for img in imgs:
    pths = os.path.join(root,"trainsmall/RGB/",img)
    imgpaths.append(pths)
for CMimg in CMimgs:
    pths = os.path.join(root,"trainsmall/CM_gt/",CMimg)
    CMpaths.append(pths)
for EMimg in EMimgs:
    pths = os.path.join(root,"trainsmall/EM_gt/",EMimg)
    EMpaths.append(pths)    
dict={'images' : imgpaths, 'EM' : EMpaths, 'CM' : CMpaths}
df = pd.DataFrame(data = dict)
df.to_json("traindatasmall.json")