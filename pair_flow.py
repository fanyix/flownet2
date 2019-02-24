import torch
import numpy as np
import argparse
import sys
from skimage.transform import resize
import math
import flow_vis
from scipy.misc import imsave 


from models import FlowNet2 #the path is depended on where you create this module
from utils.frame_utils import read_gen #the path is depended on where you create this module 

def compute_size(h, w, divisible):
    new_h = int(math.ceil(float(h) / divisible) * divisible)
    new_w = int(math.ceil(float(w) / divisible) * divisible)
    return new_h, new_w

if __name__ == '__main__':
    #obtain the necessary args for construct the flownet framework
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()
    
    # params
    args.model_path = '/disk1/fanyi-data/flownet2/FlowNet2_checkpoint.pth.tar'
    img_path1 = '/disk1/fanyi-data/DAVIS/DAVIS/JPEGImages/480p/tennis-vest/00022.jpg'
    img_path2 = '/disk1/fanyi-data/DAVIS/DAVIS/JPEGImages/480p/tennis-vest/00023.jpg'
    flow_file_path = '/home/fanyix/code/flownet2-pytorch/work/flow.flo'
    flow_img_path = '/home/fanyix/code/flownet2-pytorch/work/flow.png'

    #initial a Net
    net = FlowNet2(args).cuda()
    pretrained_dict = torch.load(args.model_path)['state_dict']
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    net.cuda()
    net.eval()
    
    #load the image pair, you can find this operation in dataset.py
    pim1 = read_gen(img_path1)
    pim2 = read_gen(img_path2)
    orig_h, orig_w = pim1.shape[0], pim1.shape[1]
    h, w = compute_size(orig_h, orig_w, divisible=64)
    pim1 = resize(pim1, (h, w), anti_aliasing=True) * 255
    pim2 = resize(pim2, (h, w), anti_aliasing=True) * 255
    
    images = [pim1, pim2]
    images = np.array(images).transpose(3, 0, 1, 2)
    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

    #process the image pair to obtian the flow 
    result = net(im).squeeze()
    
    #save flow, I reference the code in scripts/run-flownet.py in flownet2-caffe project 
    def writeFlow(name, flow):
        f = open(name, 'wb')
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)
        f.flush()
        f.close()

    flo = result.data.cpu().numpy().transpose(1, 2, 0)
    # writeFlow(flow_file_path, flo)
    # apply the coloring (for OpenCV, set convert_to_bgr=True)
    flow_color = flow_vis.flow_to_color(flo, convert_to_bgr=False)
    imsave(flow_img_path, flow_color)
    
    
    
    
    
    
    
    