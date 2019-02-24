
import sys
import cv2
import torch
import numpy as np
from math import ceil
from torch.autograd import Variable
from scipy.ndimage import imread
import models
import flow_vis
from scipy.misc import imsave 
import glob
import os
import futils 
import pytorch_utils as pu
import argparse

from models import FlowNet2 #the path is depended on where you create this module
from utils.frame_utils import read_gen #the path is depended on where you create this module 

def compute_size(h, w, divisible):
    new_h = int(ceil(float(h) / divisible) * divisible)
    new_w = int(ceil(float(w) / divisible) * divisible)
    return new_h, new_w

# params
model_fn = '/disk1/fanyi-data/flownet2/models/FlowNet2_checkpoint.pth.tar'
data_dir = '/disk3/fanyi-data/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000'
write_dir = '/disk1/fanyi-data/flownet2/output'

parser = argparse.ArgumentParser()
parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument("--rgb_max", type=float, default=255.)
args = parser.parse_args()

videos = os.listdir(data_dir)
for video_dir in videos:
    
    # write dir
    video_write_dir = os.path.join(write_dir, video_dir)
    
    # get all images
    img_list = glob.glob(os.path.join(data_dir, video_dir + '/*'))
    postfix = os.path.basename(img_list[0]).split('.')[-1]
    img_idx = np.array([int(os.path.basename(x).rstrip('.' + postfix)) for x in img_list])
    img_idx = np.argsort(img_idx)
    img_list = np.array(img_list)[img_idx]
    img_list = img_list.tolist()
    
    #initial a Net
    net = FlowNet2(args).cuda()
    pretrained_dict = torch.load(model_fn)['state_dict']
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    net.cuda()
    net.eval()
    
    # init the html info
    im_paths, captions = futils.initHTML(3, len(img_list)-1)
    
    # loop over image list
    for cur_idx in range(len(img_list)-1):
        next_idx = cur_idx + 1
        
        cur_im, next_im = imread(img_list[cur_idx]), imread(img_list[next_idx])
        
        h, w = compute_size(cur_im.shape[0], cur_im.shape[1], divisible=64)
        cur_im = cv2.resize(cur_im, (w, h))
        next_im = cv2.resize(next_im, (w, h))
        images = [cur_im, next_im]
        images = np.array(images).transpose(3, 0, 1, 2)
        im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()
        result = net(im).squeeze()
        flo = result.data.cpu().numpy().transpose(1, 2, 0)
        
        # get the filename for writing
        flow_img_fn = os.path.join(video_write_dir, 'imgs', 'flow_' + os.path.basename(img_list[cur_idx]))
        cur_img_fn = os.path.join(video_write_dir, 'imgs', 'frm_' + os.path.basename(img_list[cur_idx]))
        next_img_fn = os.path.join(video_write_dir, 'imgs', 'frm_' + os.path.basename(img_list[next_idx]))
        if not os.path.isdir(os.path.split(flow_img_fn)[0]):
            os.makedirs(os.path.split(flow_img_fn)[0])
        
        # save the flow file
        # writeFlowFile(flow_fn, flo)
        
        # apply the coloring (for OpenCV, set convert_to_bgr=True)
        flow_color = flow_vis.flow_to_color(flo, convert_to_bgr=False)
        imsave(flow_img_fn, flow_color)
        imsave(cur_img_fn, cur_im)
        if cur_idx == len(img_list)-2:
            imsave(next_img_fn, next_im)
        
        # add to html
        im_paths[0][cur_idx] = futils.relative_path(video_write_dir, cur_img_fn)
        im_paths[1][cur_idx] = futils.relative_path(video_write_dir, next_img_fn)
        im_paths[2][cur_idx] = futils.relative_path(video_write_dir, flow_img_fn)    
        captions[0][cur_idx] = 'Frame %d' % cur_idx
        captions[1][cur_idx] = 'Frame %d' % next_idx
        captions[2][cur_idx] = 'Flow %d->%d' % (cur_idx, next_idx)
        
        # logging
        print('%d/%d' % (cur_idx, len(img_list)-1))
    
    # write html
    html_file = os.path.join(video_write_dir, 'vis.html')
    futils.writeHTML(html_file, im_paths, captions, 200, 200)
    print('Done.')

