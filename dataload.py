import pandas as pd
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

imgs = os.listdir('test/RGB')
CMimgs = os.listdir("test/CM_gt")
EMimgs = os.listdir("test/EM_gt")
imgpaths = []
CMpaths = []
EMpaths = []
root=""
#abspth = os.path.abspath(root)
for img in imgs:
    pths = os.path.join(root,"test/RGB/",img)
    imgpaths.append(pths)
for CMimg in CMimgs:
    pths = os.path.join(root,"test/CM_gt/",CMimg)
    CMpaths.append(pths)
for EMimg in EMimgs:
    pths = os.path.join(root,"test/EM_gt/",EMimg)
    EMpaths.append(pths)    
dict={'images' : imgpaths, 'EM' : EMpaths, 'CM' : CMpaths}
df = pd.DataFrame(data = dict)
df.to_json("testdata.json")