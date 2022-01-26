# -*- coding: utf-8 -*-
# /usr/bin/env python3
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
#      France
#
# This software is governed by the CeCILL license version 2 under
# French law and abiding by the rules of distribution of free software.
# You can  use, modify and/or redistribute the software under the
# terms of the CeCILL license version 2 as circulated by CEA, CNRS
# and INRIA at the following URL "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.

from collections import OrderedDict
import numpy as np
import torch
import pandas as pd
from torch.autograd import Variable
import torch.nn as nn

from deep_folding.utils.pytorchtools import EarlyStopping



class InpaintAE(nn.Module):
    """ AE class with an inpainting objective
    """
    def __init__(self, in_shape, n_latent, depth):
        """
        Args:
            in_shape: tuple, input shape
            n_latent: int, latent space size
            depth: int, depth of the model

        """
        super().__init__()
        self.in_shape = in_shape
        self.n_latent = n_latent
        c,h,w,d = in_shape
        self.depth = depth
        self.z_dim_h = h//2**depth # receptive field downsampled 2 times
        self.z_dim_w = w//2**depth
        self.z_dim_d = d//2**depth

        modules_encoder = []
        for step in range(depth):
            in_channels = 1 if step == 0 else out_channels
            out_channels = 16 if step == 0  else 16 * (2**step)
            modules_encoder.append(('conv%s' %step, nn.Conv3d(in_channels, out_channels,
                    kernel_size=3, stride=1, padding=1)))
            modules_encoder.append(('norm%s' %step, nn.BatchNorm3d(out_channels)))
            modules_encoder.append(('LeakyReLU%s' %step, nn.LeakyReLU()))
            modules_encoder.append(('conv%sa' %step, nn.Conv3d(out_channels, out_channels,
                    kernel_size=4, stride=2, padding=1)))
            modules_encoder.append(('norm%sa' %step, nn.BatchNorm3d(out_channels)))
            modules_encoder.append(('LeakyReLU%sa' %step, nn.LeakyReLU()))
        self.encoder = nn.Sequential(OrderedDict(modules_encoder))

        self.z_space = nn.Linear(64 * self.z_dim_h * self.z_dim_w* self.z_dim_d, n_latent)
        self.z_develop = nn.Linear(n_latent, 64 *self.z_dim_h * self.z_dim_w* self.z_dim_d)

        modules_decoder = []
        for step in range(depth-1):
            in_channels = out_channels
            out_channels = in_channels // 2
            ini = 1 if step==0 else 0
            modules_decoder.append(('convTrans3d%s' %step, nn.ConvTranspose3d(in_channels,
                        out_channels, kernel_size=2, stride=2, padding=0, output_padding=(ini,0,0))))
            modules_decoder.append(('normup%s' %step, nn.BatchNorm3d(out_channels)))
            modules_decoder.append(('ReLU%s' %step, nn.ReLU()))
            modules_decoder.append(('convTrans3d%sa' %step, nn.ConvTranspose3d(out_channels,
                        out_channels, kernel_size=3, stride=1, padding=1)))
            modules_decoder.append(('normup%sa' %step, nn.BatchNorm3d(out_channels)))
            modules_decoder.append(('ReLU%sa' %step, nn.ReLU()))
        modules_decoder.append(('convtrans3dn', nn.ConvTranspose3d(16, 1, kernel_size=2,
                        stride=2, padding=0)))
        modules_decoder.append(('conv_final', nn.Conv3d(1, 2, kernel_size=1, stride=1)))
        self.decoder = nn.Sequential(OrderedDict(modules_decoder))
        self.weight_initialization()

    def weight_initialization(self):
        """
        Initializes model parameters according to Gaussian Glorot initialization
        """
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose3d) or isinstance(module, nn.Conv3d):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def encode(self, x):
        x = self.encoder(x)
        x = nn.functional.normalize(x, p=2)
        x = x.view(x.size(0), -1)
        encoded = self.z_space(x)
        return encoded

    def decode(self, z):
        out = self.z_develop(z)
        out = out.view(z.size(0), 16 * 2**(self.depth-1), self.z_dim_h, self.z_dim_w, self.z_dim_d)
        out = self.decoder(out)
        return out

    def forward(self, x):
        encoded = self.encode(x)
        out = self.decode(z)
        return out, encoded


def inpaint_loss(output, input, mean, logvar, loss_func, kl_weight):
    recon_loss = loss_func(output, input)
    kl_loss = -0.5 * torch.sum(-torch.exp(logvar) - mean**2 + 1. + logvar)
    return recon_loss, kl_loss, recon_loss + kl_weight * kl_loss
