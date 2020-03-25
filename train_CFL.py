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
#import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

"""
def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y
"""  

def ce_loss(pred, gt):
    '''
    pred and gt have to be the same dimensions of N x C x H x W
    weighting factors are calculated according to the CFL paper
    where W per image (single channel) in minibatch = total number of pixels/ 
    number of positive or negative labels in that image 
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    
    pos_weights = (torch.numel(pred[0][0]))/(torch.sum((pos_inds==1.).float(),dim=(1,2,3)))
    neg_weights = (torch.numel(pred[0][0]))/(torch.sum((neg_inds==1.).float(),dim=(1,2,3)))
    
    loss = 0

    pos_loss = pos_weights.view(-1,1,1,1) * (gt * -torch.log(pred))
    neg_loss = neg_weights.view(-1,1,1,1) * ((1 - gt)*(-torch.log(1-pred)))

    
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    loss = pos_loss + neg_loss

    
    return loss
    
class CELoss(nn.Module):
  '''nn.Module warpper for custom CE loss'''
  def __init__(self):
    super(CELoss, self).__init__()
    self.ce_loss = ce_loss

  def forward(self, out, target):
    return self.ce_loss(out, target)


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

def map_loss(inputs, EM_gt,CM_gt,criterion):
    '''
    function to calculate total loss according to CFL paper
    '''
    EMLoss=0.
    CMLoss=0.
    for key in inputs:
        output=torch.sigmoid(inputs[key])
        EM=F.interpolate(EM_gt,size=(output.shape[-2],output.shape[-1]),mode='bilinear',align_corners=True)
        CM=F.interpolate(CM_gt,size=(output.shape[-2],output.shape[-1]),mode='bilinear',align_corners=True)
        edges,corners =torch.chunk(output,2,dim=1)
        #edges,corners = torch.squeeze(edges,dim=1), torch.squeeze(corners,dim=1) 
        #EM,CM = torch.squeeze(EM,dim=1), torch.squeeze(CM,dim=1)
        EMLoss += criterion(edges,EM)
        CMLoss += criterion(corners,CM)        
    return EMLoss, CMLoss

def _train(args):
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

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    logger.info("Device Type: {}".format(device))

    logger.info("Loading SUN360 dataset")
    transform = transforms.Compose(
        [transforms.Resize((224,224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    target_transform = transforms.Compose([transforms.Resize((224,224)),
                                           transforms.ToTensor()])     

    trainset = SUN360Dataset("imagedata.json",transform = transform, target_transform = target_transform)
    train_loader = DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers)
    """
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False,
                                           download=False, transform=transform)
    test_loader = DataLoader(testset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers)
    """                                          

    logger.info("Model loaded")
    model = EfficientNet.from_name('efficientnet-b0',conv_type='Equi')

    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)

    criterion = CELoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(0, args.epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs
            inputs, EM , CM = data
            inputs, EM, CM = inputs.to(device), EM.to(device), CM.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            EMLoss, CMLoss = map_loss(outputs,EM,CM,criterion)
            loss = EMLoss + CMLoss
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')
    return _save_model(model, args.model_dir)


def _save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


def model_fn(model_dir):
    logger.info('model_fn')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EfficientNet.from_name('efficient-b0',conv_type='Equi')
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
    parser.add_argument('--epochs', type=int, default=1, metavar='E',
                        help='number of total epochs to run (default: 1)')
    parser.add_argument('--batch_size', type=int, default=4, metavar='BS',
                        help='batch size (default: 4)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--model-dir', type=str, default="")
    #parser.add_argument('--dist_backend', type=str, default='gloo', help='distributed backend (default: gloo)')

    #env = sagemaker_containers.training_env()
    #parser.add_argument('--hosts', type=list, default=env.hosts)
    #parser.add_argument('--current-host', type=str, default=env.current_host)
    #parser.add_argument('--model-dir', type=str, default=env.model_dir)
    #parser.add_argument('--data-dir', type=str, default=env.channel_input_dirs.get('training'))
    #parser.add_argument('--num-gpus', type=int, default=env.num_gpus)

    _train(parser.parse_args())