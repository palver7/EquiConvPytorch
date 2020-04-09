from CFLPytorch.EfficientCFL import EfficientNet
import argparse
import logging
#import sagemaker_containers
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
from skimage.feature import corner_peaks, peak_local_max
from CFLPytorch.offsetcalculator import offcalc

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def evaluate(pred, gt):
    """
    if map == 'edges':
        prediction_path_list = glob.glob(os.path.join(args.results,'EM_test')+'/*.jpg')
        gt_path_list = glob.glob(os.path.join(args.dataset, 'EM_gt')+'/*.jpg')
    if map == 'corners':
        prediction_path_list = glob.glob(os.path.join(args.results,'CM_test')+'/*.jpg')
        gt_path_list = glob.glob(os.path.join(args.dataset, 'CM_gt')+'/*.jpg')
    prediction_path_list.sort()
    gt_path_list.sort()
    """

    #P, R, Acc, f1, IoU = [], [], [], [], []
    # predicted image
    #prediction = Image.open(prediction_path_list[im])
    #pred_H, pred_W = pred.shape[0], pred.shape[1]
    #prediction = torch.tensor(prediction)/255.
    

    # gt image
    #gt = Image.open(gt_path_list[im])
    #gt = gt.resize([pred_W, pred_H])
    #gt = torch.tensor(gt)/255.
    gt = (gt.ge(0.1)).int()

    th=0.1
    gtpos=gt.eq(1)
    gtneg=gt.eq(0)
    predgt=pred.gt(th)
    predle=pred.le(th)
    tp = torch.sum((gtpos & predgt).float())
    tn = torch.sum((gtneg & predle).float())
    fp = torch.sum((gtneg & predgt).float())
    fn = torch.sum((gtpos & predle).float())

    # How accurate the positive predictions are
    #P.append(tp / (tp + fp))
    P = tp / (tp + fp)
    # Coverage of actual positive sample
    #R.append(tp / (tp + fn))
    R = (tp / (tp + fn))
    # Overall performance of model
    #Acc.append((tp + tn) / (tp + tn + fp + fn))
    Acc = ((tp + tn) / (tp + tn + fp + fn))
    # Hybrid metric useful for unbalanced classes 
    #f1.append(2 * (tp / (tp + fp))*(tp / (tp + fn))/((tp / (tp + fp))+(tp / (tp + fn))))
    f1 = (2 * (tp / (tp + fp))*(tp / (tp + fn))/((tp / (tp + fp))+(tp / (tp + fn))))
    # Intersection over Union
    #IoU.append(tp / (tp + fp + fn))
    IoU = (tp / (tp + fp + fn))
      

    #return torch.mean(P), torch.mean(R), torch.mean(Acc), torch.mean(f1), torch.mean(IoU)
    return P, R, Acc, f1, IoU

class SUN360Dataset(Dataset):
    

    def __init__(self, json_file, transform=None, target_transform=None):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            transform (callable, optional): Optional transform to be applied
                on an image.
            target_file (callable, optional): Optional transform to be applied
                on a map (edge and corner).    
        """
        self.images_data = pd.read_json(json_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images_data.iloc[idx, 0]                        
        EM_name = self.images_data.iloc[idx, 1]
        CM_name = self.images_data.iloc[idx, 2]
        image = Image.open(img_name)
        EM = Image.open(EM_name)
        CM = Image.open(CM_name)
        """
        EM = np.asarray(EM)
        EM = np.expand_dims(EM, axis=2)
        CM = np.asarray(CM) 
        CM = np.expand_dims(CM, axis=2) 
        gt = np.concatenate((EM,CM),axis = 2)
        maps = Image.fromarray(gt)
        """
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            CM = self.target_transform(CM)
            EM = self.target_transform(EM)    

        return image, EM, CM


def corners_2_xy(outputs):
    output = _sigmoid(outputs['output'])
    edges,corners =torch.chunk(output,2,dim=1)
    corner1= 255* corners
    corner1[corner1>127] = 255
    corner1[corner1<127] = 0
    corner1 = torch.cat((corner1,corner1,corner1),dim=-1)
    corner1 = torch.squeeze(corner1)
    array = corner1.detach().cpu().numpy().astype(np.uint8)
    local_peaks = corner_peaks(array, min_distance=5, threshold_rel=0.5, indices=True)
    local_peaks = np.array(local_peaks, dtype=np.float64)
    height, width = array.shape
    width /=3
    col1m = (local_peaks[:,1]>=width) & (local_peaks[:,1]<2*width)
    peaks = local_peaks[col1m] 
    peaks[:,0]/=height
    peaks[:,1]-= width
    peaks[:,1]/= width
    return peaks 


def map_predict(outputs, EM_gt,CM_gt):
    '''
    function to calculate total loss according to CFL paper
    '''
    output=_sigmoid(outputs['output'])
    EM=F.interpolate(EM_gt,size=(output.shape[-2],output.shape[-1]),mode='bilinear',align_corners=True)
    CM=F.interpolate(CM_gt,size=(output.shape[-2],output.shape[-1]),mode='bilinear',align_corners=True)
    edges,corners =torch.chunk(output,2,dim=1)
    #edges,corners = torch.squeeze(edges,dim=1), torch.squeeze(corners,dim=1) 
    #EM,CM = torch.squeeze(EM,dim=1), torch.squeeze(CM,dim=1)
    P_e, R_e, Acc_e, f1_e, IoU_e = evaluate(edges,EM)
    print('EDGES: IoU: ' + str('%.3f' % IoU_e) + '; Accuracy: ' + str('%.3f' % Acc_e) + '; Precision: ' + str('%.3f' % P_e) + '; Recall: ' + str('%.3f' % R_e) + '; f1 score: ' + str('%.3f' % f1_e))
    P_c, R_c, Acc_c, f1_c, IoU_c = CMMetric=evaluate(corners, CM)
    print('CORNERS: IoU: ' + str('%.3f' % IoU_c) + '; Accuracy: ' + str('%.3f' % Acc_c) + '; Precision: ' + str('%.3f' % P_c) + '; Recall: ' + str('%.3f' % R_c) + '; f1 score: ' + str('%.3f' % f1_c))

def _test(args):
    """
    is_distributed = len(args.hosts) > 1 and args.dist_backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=args.dist_backend, rank=host_rank, world_size=world_size)
        logger.info(
            'Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
                args.dist_backend,
                dist.get_world_size()) + 'Current host rank is {}. Using cuda: {}. Number of gpus: {}'.format(
                dist.get_rank(), torch.cuda.is_available(), args.num_gpus))
    """            

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info("Device Type: {}".format(device))
    img_size=EfficientNet.get_image_size(args.model_name)
    logger.info("Loading SUN360 dataset")
    transform = transforms.Compose(
        [transforms.Resize((img_size,img_size)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    target_transform = transforms.Compose([transforms.Resize((img_size,img_size)),
                                           transforms.ToTensor()])     

    testset = SUN360Dataset("testdata.json",transform = transform, target_transform = target_transform)
    test_loader = DataLoader(testset, batch_size=args.batch_size,
                                               shuffle=False, num_workers=args.workers)
                                              
    layerdict, offsetdict = offcalc(args.batch_size)
    logger.info("Model loaded")
    model = EfficientNet.from_pretrained(args.model_name,conv_type='Std')
    model.load_state_dict(torch.load("model.pth"))

    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    with torch.no_grad():
        model = model.to(device)
        for i, data in enumerate(test_loader):
            # get the inputs
            inputs, EM , CM = data
            inputs, EM, CM = inputs.to(device), EM.to(device), CM.to(device)
            model.eval()
            outputs = model(inputs)
            detection= corners_2_xy(outputs)
            print(len(detection))
            #map_predict(outputs,EM,CM)
        
    print('Finished Testing')
    


def _save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


def model_fn(model_dir):
    logger.info('model_fn')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EfficientNet.from_pretrained('efficient-b0',conv_type='Equi')
    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=2, metavar='W',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--batch_size', type=int, default=1, metavar='BS',
                        help='batch size (default: 1)')
    parser.add_argument('--model-dir', type=str, default="")
    parser.add_argument('--model-name', type=str, default="efficientnet-b0")
    #parser.add_argument('--dist_backend', type=str, default='gloo', help='distributed backend (default: gloo)')

    #env = sagemaker_containers.training_env()
    #parser.add_argument('--hosts', type=list, default=env.hosts)
    #parser.add_argument('--current-host', type=str, default=env.current_host)
    #parser.add_argument('--model-dir', type=str, default=env.model_dir)
    #parser.add_argument('--data-dir', type=str, default=env.channel_input_dirs.get('training'))
    #parser.add_argument('--num-gpus', type=int, default=env.num_gpus)

    _test(parser.parse_args())