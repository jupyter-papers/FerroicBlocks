"""
Created on Tue Feb 12 14:33:30 2019

@author: ziatdinovmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class conv2dblock(nn.Module):
    '''Creates a block consisting of convolutional
       layer, leaky relu and (optionally) dropout 
       and batch normalization '''
    
    def __init__(self, input_channels, output_channels,
                 kernel_size = 3, stride = 1, padding = 1,
                 use_batchnorm = False, lrelu_a = 0.01,
                 dropout_ = 0):
        '''Instantiates parameters of this block''' 
        
        super(conv2dblock, self).__init__()

        block = [] 
        block.append(nn.Conv2d(input_channels, 
                               output_channels,
                               kernel_size = kernel_size,
                               stride = stride,
                               padding = padding))
        if dropout_ > 0:
            block.append(nn.Dropout(dropout_))
        
        block.append(nn.LeakyReLU(negative_slope = lrelu_a))
        
        if use_batchnorm:
            block.append(nn.BatchNorm2d(output_channels))            
       
        self.block = nn.Sequential(*block)

        
    def forward(self, x):
        '''Forward path for this block'''
        
        output = self.block(x)
              
        return output
    

class dilation_block(nn.Module):
    '''Creates a block with dilated convolutional 
       layers (aka atrous convolutions)'''
    
    def __init__(self, input_channels, output_channels,
                 dilation_values, padding_values,
                 kernel_size = 3, stride = 1, lrelu_a = 0.01,
                 use_batchnorm = False, dropout_ = 0):
        '''Instantiates parameters of this block'''
        
        super(dilation_block, self).__init__()
        
        atrous_module = []
        for idx, (dil, pad) in enumerate(zip(dilation_values, padding_values)):
            input_channels = output_channels if idx > 0 else input_channels
            atrous_module.append(nn.Conv2d(input_channels,
                                              output_channels,
                                              kernel_size = kernel_size,
                                              stride = stride,
                                              padding = pad,
                                              dilation = dil,
                                              bias = True))
            if dropout_ > 0:
                atrous_module.append(nn.Dropout(dropout_))
                
            atrous_module.append(nn.LeakyReLU(negative_slope = lrelu_a))
            
            if use_batchnorm:
                atrous_module.append(nn.BatchNorm2d(output_channels))
                
        self.atrous_module = nn.Sequential(*atrous_module)

    def forward(self, x):
        '''Forward path for this block'''
        
        atrous_layers = []
        for conv_layer in self.atrous_module:
            x = conv_layer(x)
            atrous_layers.append(x.unsqueeze(-1))
        
        return torch.sum(torch.cat(atrous_layers, dim=-1), dim = -1)
        

class upsample_block(nn.Module):
    '''Defines upsampling block.
       Upsampling can be performed either with
       bilinear interpolation followed by 1-by-1
       convolution or with a transposed convolution'''
    
    def __init__(self, input_channels, output_channels,
                mode = 'interpolate', kernel_size = 1,
                stride = 1, padding = 0):
        '''Initiates parameters of this block'''
        
        super(upsample_block, self).__init__()
                
        self.mode = mode
       
        self.conv = nn.Conv2d(
            input_channels, output_channels, 
            kernel_size = kernel_size,
            stride = stride, padding = padding)
        
        self.conv_t = nn.ConvTranspose2d(
            input_channels, output_channels,
            kernel_size=2, stride=2, padding = 0)
   
    def forward(self, x):
        '''Defines a forward path'''
        
        
        if self.mode == 'interpolate':
        
            x = F.interpolate(
                x, scale_factor = 2,
                mode = 'bilinear', align_corners=False)
            
            return self.conv(x)

        return self.conv_t(x)
    
    
class atomsegnet(nn.Module):
    '''Builds  a fully convolutional
       neural network model'''
    
    def __init__(self, nb_classes):
        super(atomsegnet, self).__init__()
        '''Initiates model parameters'''  
        
        self.c1 = conv2dblock(1, 16)
        
        self.c2 = nn.Sequential(conv2dblock(16, 32),
                                conv2dblock(32, 32))
        
        self.c3 = nn.Sequential(conv2dblock(32, 64,
                                dropout_ = 0.3),
                                conv2dblock(64, 64,
                                dropout_ = 0.3))
        
        self.bn = dilation_block(64, 128,
                                 dilation_values = [2, 4, 6],
                                 padding_values = [2, 4, 6],
                                 dropout_ = 0.5)
        
        self.upsample_block1 = upsample_block(128, 64)
        
        self.c4 = nn.Sequential(conv2dblock(64+64, 64,
                                dropout_ = 0.3),
                                conv2dblock(64, 64,
                                dropout_ = 0.3))
        
        self.upsample_block2 = upsample_block(64, 32)
        
        self.c5 = nn.Sequential(conv2dblock(32+32, 32),
                                conv2dblock(32, 32))
        
        self.upsample_block3 = upsample_block(32, 16)
        
        self.c6 = conv2dblock(16+16, 16)
        
        self.px = nn.Conv2d(16, nb_classes, kernel_size = 1,
                            stride = 1, padding = 0)
               
                                
       
    def forward(self, x):
        '''Defines a forward path'''
        
        # Contracting path
        c1 = self.c1(x)
        d1 = F.max_pool2d(c1, kernel_size=2, stride=2)      
        
        c2 = self.c2(d1)
        d2 = F.max_pool2d(c2, kernel_size=2, stride=2)  
        
        c3 = self.c3(d2)
        d3 = F.max_pool2d(c3, kernel_size=2, stride=2) 
        
        # Atrous convolutions
        bn = self.bn(d3)
        
        # Expanding path
        u3 = self.upsample_block1(bn)
        u3 = torch.cat([c3, u3], dim = 1)
        u3 = self.c4(u3)
        
        
        u2 = self.upsample_block2(u3)
        u2 = torch.cat([c2, u2], dim = 1)
        u2 = self.c5(u2)
        
        u1 = self.upsample_block3(u2)
        u1 = torch.cat([c1, u1], dim = 1)
        u1 = self.c6(u1)

        # pixel-wise classification
        px = self.px(u1)
        output = F.log_softmax(px, dim = 1)
        
        return output
