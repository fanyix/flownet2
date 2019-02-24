
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from os import path


def last_conv_out_dim(model):
    out_dim = -1
    for i in range(len(model)-1, -1, -1):
        if hasattr(model[i], 'out_channels'):
            out_dim = model[i].out_channels
            break
    return out_dim

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    """
    from torchvision.models.inception import inception_v3
    import numpy as np
    from scipy.stats import entropy
    
    N = imgs.shape[0]

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    dtype = torch.cuda.FloatTensor

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for batch_start in range(0, N, batch_size):
        batch = imgs[batch_start:min(batch_start+batch_size,N), :, :, :]
        batch = dtype(batch)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]
        preds[batch_start:batch_start+batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def monkey_patch_rebuild_tensor_v2():
    # Monkey-patch because I trained with a newer version.
    # This can be removed once PyTorch 0.4.x is out.
    # See https://discuss.pytorch.org/t/question-about-rebuild-tensor-v2/14560
    import torch._utils
    try:
        torch._utils._rebuild_tensor_v2
    except AttributeError:
        def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
            tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
            tensor.requires_grad = requires_grad
            tensor._backward_hooks = backward_hooks
            return tensor
        torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

def create_one_hot(index, K):
    index_flat = index.data.view(-1)
    index_scatter = torch.FloatTensor(index_flat.size(0), K).zero_()
    for idx in range(index_scatter.size(0)):
        index_scatter[idx, index_flat[idx]] = 1.0
    return index_scatter

def pdist(mp1, mp2):
    m = mp1.size(0)
    n = mp2.size(0)
    d = mp1.size(1)
    mmp1 = mp1.view(m, 1, d).expand(m, n, d)
    mmp2 = mp2.view(1, n, d).expand(m, n, d)    
    mm = torch.sum((mmp1-mmp2)**2,2).view(m, n)
    mm = torch.sqrt(mm)
    return mm
        
def RGB2Gray(RGB):
    # Y' = 0.299 R + 0.587 G + 0.114 B 
    gray = RGB[:, 0, :, :] * 0.299 + RGB[:, 1, :, :] * 0.587 + RGB[:, 2, :, :] * 0.114
    gray = gray.view(RGB.size(0), 1, RGB.size(2), RGB.size(3))
    gray = gray.expand_as(RGB).contiguous()
    return gray

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
    return init_fun

def norm_uniform_to_vgg_input(data):
    output = data.clone()
    uniform_mean = (0.5, 0.5, 0.5)
    uniform_std = (0.5, 0.5, 0.5)
    vgg_mean = (0.485, 0.456, 0.406)
    vgg_std = (0.229, 0.224, 0.225)
    for idx in range(3):
        output[:, idx, :, :] = output[:, idx, :, :] * uniform_std[idx] + uniform_mean[idx]
        output[:, idx, :, :] = (output[:, idx, :, :] - vgg_mean[idx]) / vgg_std[idx]
    return output

def norm_uniform(data):
    output = data.clone()
    uniform_mean = (0.5, 0.5, 0.5)
    uniform_std = (0.5, 0.5, 0.5)
    for idx in range(3):
        output[:, idx, :, :] = (output[:, idx, :, :] - uniform_mean[idx]) / uniform_std[idx] 
    return output

def recover_norm_uniform(data):
    output = data.clone()
    uniform_mean = (0.5, 0.5, 0.5)
    uniform_std = (0.5, 0.5, 0.5)
    for idx in range(3):
        output[:, idx, :, :] = output[:, idx, :, :] * uniform_std[idx] + uniform_mean[idx]
    return output

def save_unit(network, save_dir, network_label, epoch_label, gpu_ids):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = path.join(save_dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if len(gpu_ids) and torch.cuda.is_available():
        network.cuda()

class NetAnalyzer():
    def __init__(self):
        self.record = []
        
    def print_weight_grad(self, m):
        if hasattr(m, 'weight') and m.weight is not None:
            # get the module name
            classname = m.__str__()
            w = torch.norm(m.weight.data)
            if hasattr(m.weight, 'grad') and m.weight.grad is not None:
                g = torch.norm(m.weight.grad.data)
            else:
                g = 0
            gw_ratio = g / (w+1e-8)
            struct = {'name':classname, 'w':w, 'g':g, 'gw_ratio':gw_ratio}
            self.record.append(struct) 
            
class Reshape(nn.Module):
    def __init__(self, size=None):
        super(Reshape, self).__init__()
        self.size = size
        
    def forward(self, x):
        if self.size is None:
            y = x.view(x.size(0), -1)
        else:
            y = x.view(self.size)
        return y

class DbTunnel(nn.Module):
    def __init__(self):
        super(DbTunnel, self).__init__()
    def forward(self, x):
        return x
            
class FlipChannel(nn.Module):
    def __init__(self, dim):
        super(FlipChannel, self).__init__(dim)
        self.dim = dim
    def forward(self, x):
        N = x.size(self.dim)
        index = torch.cat([torch.range(N/2, N), torch.range(0, N/2)], 0)
        y = x.index_select(self.dim, index)
        return y

class ModelTemplate(nn.Module):
    def __init__(self):
        super(ModelTemplate, self).__init__()
        
    def save_unit(self, network, save_dir, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = path.join(save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda()
    
    # helper loading function that can be used by subclasses
    def load_unit(self, network, network_label, epoch_label, save_dir):        
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = path.join(save_dir, save_filename)        
        if not path.isfile(save_path):
            assert False, '%s not exists yet!' % save_path
        else:
            #network.load_state_dict(torch.load(save_path))
            print('Loading network %s' % save_path)
            try:
                loaded_model = torch.load(save_path)
                # print('\n')
                # print('----------- START %s -----------' % save_filename)
                # print('Total modules: %d' % len(network.state_dict().keys()))
                # print(network.state_dict().keys())
                # print('----------- END %s -----------' % save_filename)
                # print('\n')
                network.load_state_dict(loaded_model)
            except:   
                pretrained_dict = torch.load(save_path)                
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}                    
                    network.load_state_dict(pretrained_dict)
                    print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    from sets import Set
                    not_initialized = Set()
                    for k, v in pretrained_dict.items():                      
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])                            
                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)   

class ResNet50Feat(ModelTemplate):
    def __init__(self, feat_dim, cls_mode='no', pretrained_path=None, which_epoch=None, cls_num=None):
        super(ResNet50Feat, self).__init__()
        from torchvision import models
        self.feat_dim = feat_dim
        self.cls_mode = cls_mode
        self.cls_num = cls_num
        model = models.resnet50(pretrained=True)
        
        # a new version where it preserves spatial information
        model.avgpool = torch.nn.AdaptiveAvgPool2d((2, 2))
        model.fc = torch.nn.Linear(2048*2*2, self.feat_dim)
        # model.avgpool = torch.nn.AvgPool2d(2, stride=2)
        # model.fc = torch.nn.Linear(2048*4*4, self.feat_dim) # size for 256*256 input
        
        model.fc.apply(weights_init('gaussian'))
        model.cuda()
        self.model = model
        
        if self.cls_mode == 'angle':
            self.cls_linear = AngleLinear(self.feat_dim, self.cls_num)
            self.cls_linear.apply(weights_init('gaussian'))
            self.cls_linear.cuda()
        elif self.cls_mode == 'fc':
            self.cls_linear = [torch.nn.ReLU(), torch.nn.Linear(self.feat_dim, self.cls_num)]
            self.cls_linear = torch.nn.Sequential(*self.cls_linear)
            self.cls_linear.apply(weights_init('gaussian'))
            self.cls_linear.cuda()
        
        if pretrained_path is not None and pretrained_path != '':
            self.load(which_epoch, pretrained_path)
    
    def inference(self, x, batch_size=64):
        N = x.size(0)
        y = None
        
        for batch_start in range(0, N, batch_size):
            batch = x[batch_start:min(batch_start+batch_size,N), :, :, :]
            batch_size_i = batch.size(0)
            cur_preds = self.forward(batch)
            cur_preds = cur_preds.data
            if y is None:
                size = [i for i in cur_preds.size()]
                size[0] = N
                y = cur_preds.new(torch.Size(size)).zero_()
            y[batch_start:batch_start+batch_size_i] = cur_preds
        
        return y
    
    def forward(self, x):
        y = self.model.forward(x)
        if self.cls_mode != 'no':
            y = self.cls_linear.forward(y)
        return y
    
    def save(self, save_dir, which_epoch):
        self.save_unit(self.model, save_dir, 'resnet', which_epoch, [0])
        if self.cls_mode != 'no':
            self.save_unit(self.cls_linear, save_dir, 'clsLinear', which_epoch, [0])
    
    def load(self, which_epoch, pretrained_path):
        self.load_unit(self.model, 'resnet', which_epoch, pretrained_path)
        if self.cls_mode != 'no':
            self.load_unit(self.cls_linear, 'clsLinear', which_epoch, pretrained_path)

class SelfAttention(nn.Module):
    def __init__(self, feat_dim, proj_dim, gamma):
        super(SelfAttention, self).__init__()
        # init the convolution operators
        self.W_g = torch.nn.Conv2d(feat_dim, proj_dim, kernel_size=(1, 1))
        self.W_f = torch.nn.Conv2d(feat_dim, proj_dim, kernel_size=(1, 1))
        self.W_h = torch.nn.Conv2d(feat_dim, feat_dim, kernel_size=(1, 1))
        self.softmax = torch.nn.Softmax(dim=1)
        self.gamma = gamma
        
        # init the weights
        self.W_g.apply(weights_init('gaussian'))
        self.W_f.apply(weights_init('gaussian'))
        self.W_h.apply(weights_init('gaussian'))
        
        # ship to GPU
        self.W_g.cuda()
        self.W_f.cuda()
        self.W_h.cuda()
    
    def forward(self, x):
        f = self.W_f.forward(x)
        g = self.W_g.forward(x)
        h = self.W_h.forward(x)
        
        # get the dimensions
        N, feat_D, hgt, wid = x.size(0), x.size(1), x.size(2), x.size(3)
        proj_D = f.size(1)
        
        # reshape variables
        f = f.view(N, proj_D, -1).transpose(1, 2)
        g = g.view(N, proj_D, -1).transpose(1, 2)
        h = h.view(N, feat_D, -1).transpose(1, 2)
        
        o = []
        for idx in range(N):
            # compute the affinity 
            aff = torch.mm(g[idx], f[idx].transpose(0, 1))
            aff = self.softmax(aff)
            
            # synthesize the new feature
            cur_o = torch.mm(aff, h[idx])
            cur_o = cur_o.transpose(0, 1).contiguous()
            cur_o = cur_o.view(1, feat_D, hgt, wid)
            o.append(cur_o)
            
        # concatenate o
        o = torch.cat(o, 0)
        
        # generate final results
        y = self.gamma * o + (1-self.gamma) * x
        
        return y
        
        
class BCEWithLogitsIndexLoss():
    def __init__(self, mode):
        self.sigmoid = torch.nn.Sigmoid()
        self.mode = mode
    
    def forward(self, cls_prob, index, label):
        index = index.data.view(-1)
        N = index.size(0)
        loss = 0
        assert cls_prob.size(0) == N
        for idx in range(N):
            val = cls_prob[idx, index[idx]]
            if self.mode == 'bce':
                if label:
                    loss = loss - torch.log(self.sigmoid.forward(val))
                else:
                    loss = loss - torch.log(1-self.sigmoid.forward(val))
            elif self.mode == 'mse':
                if label:
                    loss = loss + (val - 1)**2
                else:
                    loss = loss + val**2
        loss = loss / N
        return loss
        
class TripletLoss():
    def __init__(self, margin):
        self.margin = margin
        self.relu = torch.nn.ReLU()
    
    def forward(self, x, pos, neg):
        assert x.size(0) == pos.size(0)
        assert x.size(0) == neg.size(0)
        # normalize feat
        x = F.normalize(x, 2, 1)
        pos = F.normalize(pos, 2, 1)
        neg = F.normalize(neg, 2, 1)
        # compute loss 
        x_pos_affinity = torch.diag(torch.mm(x, pos.transpose(1, 0))) 
        x_neg_affinity = torch.diag(torch.mm(x, neg.transpose(1, 0)))
        gap = x_neg_affinity - x_pos_affinity + self.margin
        loss = torch.mean(self.relu.forward(gap))
        return loss

class SpatialTransformer(ModelTemplate):
    def __init__(self, in_channel):
        super(SpatialTransformer, self).__init__()
        # Spatial transformer localization-network
        self.branch_net = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=7, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            nn.ReLU(True)
        )
        self.merge_net = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.AdaptiveMaxPool2d(1),
            Reshape(),
            nn.Linear(128, 3*2),
        )

        # Initialize the weights/bias with identity transformation
        self.branch_net.apply(weights_init('gaussian'))
        self.merge_net.apply(weights_init('gaussian'))
        self.merge_net[-1].weight.data.zero_()
        self.merge_net[-1].bias.data.copy_(torch.FloatTensor([1, 0, 0, 0, 1, 0]))

    # Spatial transformer network forward function
    def infer_theta(self, source, target):
        source_feat = self.branch_net(source)
        target_feat = self.branch_net(target)
        stack_feat = torch.cat([source_feat, target_feat], 1)
        theta = self.merge_net(stack_feat)
        theta = theta.view(-1, 2, 3)
        # grid = F.affine_grid(theta, x.size())
        # y = F.grid_sample(x, grid)
        return theta
            
    def transform_source(self, source, target):
        theta = self.infer_theta(source, target)
        grid = F.affine_grid(theta, source.size())
        y = F.grid_sample(source, grid)
        return y
    
    def save(self, save_dir, which_epoch):
        self.save_unit(self.branch_net, save_dir, 'branch', which_epoch, [0])
        self.save_unit(self.merge_net, save_dir, 'merge', which_epoch, [0])
    
    def load(self, which_epoch, pretrained_path):
        self.load_unit(self.branch_net, 'branch', which_epoch, pretrained_path)
        self.load_unit(self.merge_net, 'merge', which_epoch, pretrained_path)


############ Angular loss ############

def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        output = (cos_theta,phi_theta)
        return output # size=(B,Classnum,2)

class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()
        index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss
    
    
    
    
"""Spectral Normalization from https://arxiv.org/abs/1802.05957"""
def spectral_norm_batch_apply(module, name='weight', n_power_iterations=1, eps=1e-12):
    classname = module.__class__.__name__
    if (classname.find('Conv') != -1 or classname.find('Linear') != -1) \
        and hasattr(module, '_parameters') and name in module._parameters:
        print('Applying spectral normalization to %s.' % type(module))
        fn = SpectralNorm(name, n_power_iterations, eps)
        weight = module._parameters[name]
        height = weight.size(0)
        u = F.normalize(Variable(weight.data.new(height).normal_(0, 1),requires_grad=False), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_org", weight)
        module.register_buffer(fn.name, weight)
        module.register_buffer(fn.name + "_u", u)
        module.register_forward_pre_hook(fn)
        
class SpectralNorm(object):
    def __init__(self, name='weight', n_power_iterations=1, eps=1e-12):
        self.name = name
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_org')
        u = getattr(module, self.name + '_u')
        height = weight.size(0)
        weight_mat = weight.view(height, -1)
        
        # with torch.no_grad():
        #     for _ in range(self.n_power_iterations):
        #         # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
        #         # are the first left and right singular vectors.
        #         # This power iteration produces approximations of `u` and `v`.
        #         v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
        #         u = F.normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
        u_data = u.data
        weight_mat_data = weight_mat.data
        for _ in range(self.n_power_iterations):
            # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
            # are the first left and right singular vectors.
            # This power iteration produces approximations of `u` and `v`.
            v_data = F.normalize(torch.matmul(weight_mat_data.t(), u_data), dim=0, eps=self.eps)
            u_data = F.normalize(torch.matmul(weight_mat_data, v_data), dim=0, eps=self.eps)
        u = Variable(u_data, requires_grad=False)
        v = Variable(v_data, requires_grad=False)
        
        sigma = torch.dot(u, torch.matmul(weight_mat, v))
        weight = weight / sigma
        return weight, u

    def remove(self, module):
        weight = module._parameters[self.name + '_org']
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_org')
        module.register_parameter(self.name, weight)

    def __call__(self, module, inputs):
        weight, u = self.compute_weight(module)
        # setattr(module, self.name, weight)
        # setattr(module, self.name + '_u', u)
        # cur_weight = getattr(module, self.name)
        # cur_weight.data.copy_(weight.data)
        # cur_u = getattr(module, self.name + '_u')
        # cur_u.data.copy_(u.data)
        module._buffers[self.name] = weight
        module._buffers[self.name + '_u'] = u

    @staticmethod
    def apply(module, name, n_power_iterations, eps):
        fn = SpectralNorm(name, n_power_iterations, eps)
        weight = module._parameters[name]
        height = weight.size(0)
        # u = F.normalize(weight.new(height).normal_(0, 1), dim=0, eps=fn.eps)
        u = F.normalize(Variable(weight.data.new(height).normal_(0, 1),requires_grad=False), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_org", weight)
        module.register_buffer(fn.name, weight)
        module.register_buffer(fn.name + "_u", u)
        module.register_forward_pre_hook(fn)
        return fn


def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12):
    r"""Applies spectral normalization to a parameter in the given module.
    .. math::
         \mathbf{W} &= \dfrac{\mathbf{W}}{\sigma(\mathbf{W})} \\
         \sigma(\mathbf{W}) &= \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}
    Spectral normalization stabilizes the training of discriminators (critics)
    in Generaive Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.
    See `Spectral Normalization for Generative Adversarial Networks`_ .
    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectal norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
    Returns:
        The original module with the spectal norm hook
    Example::
        >>> m = spectral_norm(nn.Linear(20, 40))
        Linear (20 -> 40)
        >>> m.weight_u.size()
        torch.Size([20])
    """
    SpectralNorm.apply(module, name, n_power_iterations, eps)
    return module


def remove_spectral_norm(module, name='weight'):
    r"""Removes the spectral normalization reparameterization from a module.
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
    Example:
        >>> m = spectral_norm(nn.Linear(40, 10))
        >>> remove_spectral_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))





