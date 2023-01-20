import torch
import torch.nn as nn
from torch.autograd import Function
import torchvision.models as models
import ctlib
import numpy as np
import graph_laplacian_2 as graph_laplacian

class adj_weight(Function):
    def __init__(self, k=9):
        self.k = k
    def forward(self, x):
        return graph_laplacian.forward(x, self.k)
    
class gcn_module(nn.Module):
    def __init__(self):
        super(gcn_module, self).__init__()
        
    def forward(self, x, adj):
        out = torch.zeros_like(x)
        for i in range(x.size(0)):
            ttt = torch.mm(adj[i], torch.transpose(x[i], 0, 1))
            out[i] = torch.transpose(ttt, 0, 1)
        return out

class prj_fun(Function):
    @staticmethod
    def forward(self, input_data, weight, proj, options):
        temp = ctlib.projection(input_data, options) - proj
        intervening_res = ctlib.backprojection(temp, options, 0)
        self.save_for_backward(intervening_res, weight, options, 0)
        out = input_data - weight * intervening_res
        return out

    @staticmethod
    def backward(self, grad_output):
        intervening_res, weight, options = self.saved_tensors
        temp = ctlib.projection(grad_output, options, 0)
        temp = ctlib.backprojection(temp, options, 0)
        grad_input = grad_output - weight * temp
        temp = intervening_res * grad_output
        grad_weight = - temp.sum().view(-1)
        return grad_input, grad_weight, None, None
    
class projection(Function):
    @staticmethod
    def forward(self, input_data, options):
           # y = Ax   x = A^T y
        out = ctlib.projection(input_data, options, 0)
        self.save_for_backward(options, input_data)
        return out

    @staticmethod
    def backward(self, grad_output):
        options, input_data = self.saved_tensors
        grad_input = ctlib.backprojection(grad_output, options, 0)
        return grad_input, None
    
class back_projection(Function):
    @staticmethod
    def forward(self, input_data, options):
        #self.save_for_backward(options)   # y = Ax   x = A^T y
        out = ctlib.backprojection(input_data, options, 0)
        self.save_for_backward(options, input_data)
        return out

    @staticmethod
    def backward(self, grad_output):
        options, input_data = self.saved_tensors
        grad_input = ctlib.projection(grad_output, options, 0)
        return grad_input, None

class sigma_activation(nn.Module):
    def __init__(self, ddelta):
        super(sigma_activation, self).__init__()
        self.relu = nn.ReLU(inplace=True)      
        self.ddelta = ddelta
        self.coeff = 1.0 / (4.0 * self.ddelta)

    def forward(self, x_i):
        x_i_relu = self.relu(x_i)
        x_square = torch.mul(x_i, x_i)
        x_square *= self.coeff
        return torch.where(torch.abs(x_i) > self.ddelta, x_i_relu, x_square + 0.5*x_i + 0.25 * self.ddelta)
    
class sigma_derivative(nn.Module):
    def __init__(self, ddelta):
        super(sigma_derivative, self).__init__()
        self.ddelta = ddelta
        self.coeff2 = 1.0 / (2.0 * self.ddelta)

    def forward(self, x_i):
        x_i_relu_deri = torch.where(x_i > 0, torch.ones_like(x_i), torch.zeros_like(x_i))
        return torch.where(torch.abs(x_i) > self.ddelta, x_i_relu_deri, self.coeff2 *x_i + 0.5)

class LDA(nn.Module):
    def __init__(self, block_num, **kwargs):
        super(LDA, self).__init__()
        views = kwargs['views']
        dets = kwargs['dets']
        width = kwargs['width']
        height = kwargs['height']
        dImg = kwargs['dImg']
        dDet = kwargs['dDet']
        dAng = kwargs['dAng']
        s2r = kwargs['s2r']
        d2r = kwargs['d2r']
        binshift = kwargs['binshift']
        options = torch.Tensor([views, dets, width, height, dImg, dDet, dAng, s2r, d2r, binshift])
        
        self.options = nn.Parameter(options, requires_grad=False)
        self.thresh = nn.Parameter(torch.Tensor([0.002]), requires_grad=True)
        
        self.sigma = 10**6
        
        self.views = views
        
        channel_num = 32
        self.channel_num = channel_num
        
        self.conv0 = nn.Conv2d(1, channel_num, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1)
        
        self.deconv0 = nn.ConvTranspose2d(channel_num, 1, kernel_size=3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(channel_num, channel_num, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(channel_num, channel_num, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(channel_num, channel_num, kernel_size=3, padding=1)
        
        
        self.conv0_s = nn.Conv2d(4, channel_num, kernel_size=3, padding=1)
        self.conv1_s = nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1)
        self.conv2_s = nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1)
        self.conv3_s = nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1)
        
        self.deconv0_s = nn.ConvTranspose2d(channel_num, 4, kernel_size=3, padding=1)
        self.deconv1_s = nn.ConvTranspose2d(channel_num, channel_num, kernel_size=3, padding=1)
        self.deconv2_s = nn.ConvTranspose2d(channel_num, channel_num, kernel_size=3, padding=1)
        self.deconv3_s = nn.ConvTranspose2d(channel_num, channel_num, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True) 
        self.act = sigma_activation(0.001)
        self.deri_act = sigma_derivative(0.001)
        self.delta_ = 10**-5
        
        self.I0 = nn.Parameter(torch.Tensor([10**5]), requires_grad=True)

        self.adj_weight = adj_weight()
        
        self.block1 = gcn_module()
        
        self.kernel_size_folding = 5
        self.stride_folding = 2
        self.padding_folding = 2
        
        self.projection = projection()
        self.back_projection = back_projection()
        
        self.alpha_list = []
        for block_index in range(block_num):
            eval_func1 = "self.alpha_" + str(block_index) + " = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)"
            exec(eval_func1)
            eval_func2 = "self.alpha_list.append(" + "self.alpha_" + str(block_index) + ")"
            exec(eval_func2)
            
        self.gamma_list = []
        for block_index in range(block_num):
            eval_func1 = "self.gamma_" + str(block_index) + " = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)"
            exec(eval_func1)
            eval_func2 = "self.gamma_list.append(" + "self.gamma_" + str(block_index) + ")"
            exec(eval_func2)
            
        self.omega_list = []
        for block_index in range(block_num):
            eval_func1 = "self.omega_" + str(block_index) + " = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)"
            exec(eval_func1)
            eval_func2 = "self.omega_list.append(" + "self.omega_" + str(block_index) + ")"
            exec(eval_func2)
            
        self.beta_list = []
        for block_index in range(block_num):
            eval_func1 = "self.beta_" + str(block_index) + " = nn.Parameter(torch.Tensor([0.03]), requires_grad=True)"
            exec(eval_func1)
            eval_func2 = "self.beta_list.append(" + "self.beta_" + str(block_index) + ")"
            exec(eval_func2)
            
        self.theta_list = []
        for block_index in range(block_num):
            eval_func1 = "self.theta_" + str(block_index) + " = nn.Parameter(torch.Tensor([0.005]), requires_grad=True)"
            exec(eval_func1)
            eval_func2 = "self.theta_list.append(" + "self.theta_" + str(block_index) + ")"
            exec(eval_func2)
        
        
#    def submodule_grad_R_exact(self, lambda_, x_x4_forward):
#        
#        [x1, x2, x3, x4] = x_x4_forward
#        norm_along_d = torch.norm(x4, dim=1, keepdim=True)
#        greater = torch.sign(self.relu(norm_along_d - lambda_ * torch.abs(self.thresh)))
#        greater = greater.repeat([1, self.channel_num, 1, 1])
#        x_greater = torch.mul(greater, x4)
#        x_less = torch.div(x4 - x_greater, lambda_ * torch.abs(self.thresh))
#        x_greater = torch.nn.functional.normalize(x_greater, dim=1)
#        x4_out = x_greater + x_less
#
#        x4_dec = torch.nn.functional.conv_transpose2d(x4_out, weight=self.conv3.weight, bias=None, stride=1, padding=2)
#        x3_deri_act = self.deri_act(x3)
#        x3_dec = torch.nn.functional.conv_transpose2d(torch.mul(x3_deri_act, x4_dec), weight=self.conv2.weight, bias=None, stride=1, padding=2)
#        x2_deri_act = self.deri_act(x2)
#        x2_dec = torch.nn.functional.conv_transpose2d(torch.mul(x2_deri_act, x3_dec), weight=self.conv1.weight, bias=None, stride=1, padding=2)
#        x1_deri_act = self.deri_act(x1)
#        x1_dec = torch.nn.functional.conv_transpose2d(torch.mul(x1_deri_act, x2_dec), weight=self.conv0.weight, bias=None, stride=1, padding=2)
#        
#        return x1_dec
    
    def submodule_grad_R(self, lambda_, x_x4_forward):
        
        [x1, x2, x3, x4] = x_x4_forward
        
        thresh = lambda_ * torch.abs(self.thresh)
        norm_along_d = torch.norm(x4, dim=1, keepdim=True)
        x_denominator = torch.where(norm_along_d > thresh, norm_along_d, thresh)
        x_denominator = x_denominator.repeat([1, self.channel_num, 1, 1])
        x4_out = torch.div(x4, x_denominator)
        
        x4_dec = self.deconv3(x4_out)
        x3_deri_act = self.deri_act(x3)
        x3_dec = self.deconv2(torch.mul(x3_deri_act, x4_dec))
        x2_deri_act = self.deri_act(x2)
        x2_dec = self.deconv1(torch.mul(x2_deri_act, x3_dec))
        x1_deri_act = self.deri_act(x1)
        x1_dec = self.deconv0(torch.mul(x1_deri_act, x2_dec))
        
        return x1_dec
    
    def submodule_grad_R2(self, x_x4_forward, adj):
        
        [x1, x2, x3, x4] = x_x4_forward
        
        # graph Laplacian
        x5 = nn.functional.unfold(x4, 2, dilation=1, padding=0, stride=2)
        x6 = self.block1(x5, adj)
        x6 = nn.functional.fold(x6, 256, 2, dilation=1, padding=0, stride=2)
        
        x4_dec = self.deconv3(x6)
        x3_deri_act = self.deri_act(x3)
        x3_dec = self.deconv2(torch.mul(x3_deri_act, x4_dec))
        x2_deri_act = self.deri_act(x2)
        x2_dec = self.deconv1(torch.mul(x2_deri_act, x3_dec))
        x1_deri_act = self.deri_act(x1)
        x1_dec = self.deconv0(torch.mul(x1_deri_act, x2_dec))
        
        return x1_dec
    
    def submodule_R(self, x_x4_forward):

        x4 = x_x4_forward[-1]    
        norm_out = torch.norm(x4, dim=1, keepdim=False)
        norm_out = torch.norm(norm_out, p=1, dim=[1, 2], keepdim=False)
        
        return norm_out
    
    def x4_forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(self.act(x1))
        x3 = self.conv2(self.act(x2))
        x4 = self.conv3(self.act(x3))
        return [x1, x2, x3, x4]
        
    def x4_forward_sinogram(self, x):                                 # 1024 * 512    -> 512 * 256 * 4
        x = self.projection.apply(x, self.options)
        x = nn.functional.unfold(x, 2, dilation=1, padding=0, stride=2)     # 
        x = x.view(-1, 4, self.views // 2, 512 // 2)
        
        x1 = self.conv0_s(x)
        x2 = self.conv1_s(self.act(x1))
        x3 = self.conv2_s(self.act(x2))
        x4 = self.conv3_s(self.act(x3))
        return [x1, x2, x3, x4]
        
    def submodule_grad_sinogram(self, x_x4_forward, adj):
        
        [x1, x2, x3, x4] = x_x4_forward
        

        
        # graph Laplacian
        x5 = nn.functional.unfold(x4, self.kernel_size_folding, dilation=1, padding=self.padding_folding, stride=self.stride_folding)
        

        
        x6 = self.block1(x5, adj)
        x6 = nn.functional.fold(x6, (self.views // self.stride_folding, 512 // self.stride_folding), self.kernel_size_folding, dilation=1, padding=self.padding_folding, stride=self.stride_folding)  # 1024 * 512 * channel   # 256 * 256
        
        
        x4_dec = self.deconv3_s(x6)
        x3_deri_act = self.deri_act(x3)
        x3_dec = self.deconv2_s(torch.mul(x3_deri_act, x4_dec))
        x2_deri_act = self.deri_act(x2)
        x2_dec = self.deconv1_s(torch.mul(x2_deri_act, x3_dec))
        x1_deri_act = self.deri_act(x1)
        x1_dec = self.deconv0_s(torch.mul(x1_deri_act, x2_dec))
        
        x1_dec = x1_dec.view(-1, 4, self.views // 2 * 512 // 2)
        
        x1_dec = nn.functional.fold(x1_dec, (self.views, 512), 2, dilation=1, padding=0, stride=2)
        
        x1_dec = self.back_projection.apply(x1_dec, self.options)
        
        return x1_dec
        
        
    def function_value(self, x, proj, x_x4_forward):
        
        ff_ = torch.norm(ctlib.projection(x, self.options) - proj, p = 2, dim=[1, 2, 3], keepdim=False)
        f_ = 0.5 * ff_ * ff_ + self.submodule_R(x_x4_forward)
        return torch.mean(f_, dim=0, keepdim=False)
    
    def submodule_datafedality2(self, w_k, x, c):
        b = torch.log(self.I0) - ctlib.projection(x, self.options, 0)
        h_deri_w = w_k - c - b - torch.log(w_k) - torch.div(0.5, w_k)
        h_deri2_w = 1 - torch.div(1.0, w_k) + torch.div(0.5, torch.square(w_k))
        return torch.div(h_deri_w, h_deri2_w)
    
    def forward(self, input_data, proj, block_num):
        
        c = self.I0 * torch.exp(-proj)
        
        w = c
        
        x = input_data
        lambda_ = nn.Parameter(torch.Tensor([1.0]), requires_grad=False).cuda()
        
        # generate the graph_lap
        
        [x1, x2, x3, x4] = self.x4_forward_sinogram(x)
        
        
        # graph Laplacian
        patch1 = nn.functional.unfold(x4, self.kernel_size_folding, dilation=1, padding=self.padding_folding, stride=self.stride_folding)
        
        
        patch1 = torch.transpose(patch1, 1, 2).contiguous()
        
        adj1 = []
        for i in range(input_data.size(0)):
            adj1.append(self.adj_weight.forward(patch1[i])) # [batch, 15876, 15876]
            

        for i in range(block_num):
            # get the step size
            alpha = torch.abs(self.alpha_list[i])     
            beta = torch.abs(self.beta_list[i])
            theta = torch.abs(self.theta_list[i])
            gamma = torch.abs(self.gamma_list[i])
            omega = torch.abs(self.omega_list[i])
            
            # compute b
            # b = prj_fun.apply(x, alpha, proj, self.options)
                
            # check if lambda_ should decrease
            if i >= 1:
                lambda_ = lambda_ * 0.9
            input_second_regu = self.x4_forward_sinogram(x)
            # compute the candidate x_u and x_v
            input_to_regu = self.x4_forward(x)
            x = x - beta * self.submodule_grad_R(lambda_, input_to_regu) - theta * self.submodule_grad_sinogram(input_second_regu, adj1)
            
            # x = b - beta * self.submodule_grad_R(lambda_, input_to_regu) - theta * self.submodule_grad_R2(input_to_regu, adj1)
            
            x -= gamma * self.back_projection.apply(w, self.options)
            for _ in range(10):
                w = w - omega * self.submodule_datafedality2(w, x, c)
                
            x = self.relu(x)
            
            
        return x