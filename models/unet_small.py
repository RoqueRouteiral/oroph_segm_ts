import torch
import torch.nn as nn
import torch.nn.functional as F
import torch 
import torch.nn as nn

class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        #super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

class Small_u(nn.Module):
    def __init__(self, in_channel, n_classes=1):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(Small_u, self).__init__()
        
        #Downsampling path
        self.ec0 = self.conv_block(self.in_channel, 16, padding=1)
        self.ec1 = self.conv_block(16, 32, padding=1)
        self.pool0 = nn.MaxPool3d(2, stride=2)
        self.ec2 = self.conv_block(32, 32, padding=1)
        self.ec3 = self.conv_block(32, 64, padding=1)
        self.pool1 = nn.MaxPool3d(2, stride=2)
        self.ec4 = self.conv_block(64, 64, padding=1)
        self.ec5 = self.conv_block(64, 128, padding=1)
        self.pool2 = nn.MaxPool3d(2, stride=2)
        self.ec6 = self.conv_block(128, 128, padding=1, dropout=True)
        self.ec7 = self.conv_block(128, 256, padding=1, dropout=True)

        #Upsampling path
        self.dc9 = self.decoder(256, 256, kernel_size=2, stride=2, bias=False)
        self.dc8 = self.conv_block(128 + 256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc7 = self.conv_block(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc6 = self.decoder(128, 128, kernel_size=2, stride=2, bias=False)
        self.dc5 = self.conv_block(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc4 = self.conv_block(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc3 = self.decoder(64, 64, kernel_size=2, stride=2, bias=False)
        self.dc2 = self.conv_block(32 + 64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc1 = self.conv_block(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        #self.dc0 = self.decoder(64, n_classes, kernel_size=1, stride=1, bias=False)
        self.final = self.finalLayer(32, n_classes, kernel_size=1, stride=1, bias=False)
        

    def conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=True,dropout=False):
        if dropout:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Dropout3d(p=0.2),
                ContBatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                #nn.Dropout3d(p=0.2),
                ContBatchNorm3d(out_channels),
                nn.ReLU())
        return layer


    def decoder(self, in_channels, out_channels, kernel_size, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias))
        return layer
       
    
    def finalLayer(self, in_channels, out_channels, kernel_size, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.Sigmoid())
        return layer 
        

    def forward(self, x):
        #print(x.size())
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)

        del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)
        del e3, e4

        e5 = self.pool2(syn2)
       
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)
        del e5, e6
        d9 = torch.cat((self.dc9(e7), syn2), dim=1)
        del e7, syn2

        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        del d9, d8

        d6 = torch.cat((self.dc6(d7), syn1), dim=1)
        del d7, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0), dim=1)
        del d4, syn0

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2

        d0 = self.final(d1)
        return d0