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

import numpy as np
import pandas as pd
#from torchsummary import summary
import torchvision
from torch.utils.tensorboard import SummaryWriter

from vae import *
from deep_folding.utils.pytorchtools import EarlyStopping
from postprocess import plot_loss


def train_vae(config, trainloader, valloader, root_dir=None, curr_config=None):
    """ Trains beta-VAE for a given hyperparameter configuration
    Args:
        config: instance of class Config
        trainloader: torch loader of training data
        valloader: torch loader of validation data
        root_dir: str, directory where to save model

    Returns:
        vae: trained model
        final_loss_val
    """
    torch.manual_seed(10)
    writer = SummaryWriter(log_dir='/neurospin/dico/lguillon/distmap/CCD/'\
                                    'runs_tensorboard/runs/LEFT_ccd_75_2_seed_10',
                            comment=f"ccd_{config.n}_kl_{config.kl}")
    lr = config.lr
    #vae = VAE(config.in_shape, curr_config['n'], depth=3)
    print(config.in_shape, config.n)
    vae = VAE(config.in_shape, config.n, depth=3)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    vae.to(device)
    #summary(vae, config.in_shape)

    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    nb_epoch = config.nb_epoch
    early_stopping = EarlyStopping(patience=12, verbose=True, root_dir=root_dir)

    list_loss_train, list_loss_val = [], []

    # arrays enabling to see model reconstructions
    id_arr, phase_arr, input_arr, output_arr = [], [], [], []

    for epoch in range(config.nb_epoch):
        running_loss = 0.0
        epoch_steps = 0
        for inputs, path in trainloader:
            #print(path)
            #print("==========================TRAIN==============")
            optimizer.zero_grad()

            inputs = Variable(inputs).to(device, dtype=torch.float32)
            #target = torch.squeeze(inputs, dim=1).long()
            output, z, logvar = vae(inputs)
            """recon_loss, kl, loss = vae_loss(output, target, z,
                                    logvar, criterion,
                                    kl_weight=config.kl)"""
            recon_loss, kl, loss = vae_loss(inputs, output, z,
                                    logvar, criterion,
                                    kl_weight=config.kl)
            #output = torch.argmax(output, dim=1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_steps += 1
        # addition of one reconstruction in visualization
        #print(inputs.shape, output.shape)
        images = [inputs[0][0][20][:][:],\
                  output[0][0][20][:][:]]
        grid = torchvision.utils.make_grid(images)
        writer.add_image('inputs', images[0].unsqueeze(0), epoch)
        writer.add_image('output', images[1].unsqueeze(0), epoch)
        writer.add_scalar('Loss/train', running_loss / epoch_steps, epoch)
        writer.close()
        print("[%d] loss: %.3f" % (epoch + 1,
                                        running_loss / epoch_steps))
        list_loss_train.append(running_loss / epoch_steps)
        running_loss = 0.0

        """ Saving of reconstructions for visualization in Anatomist software """
        if epoch == nb_epoch-1:
            for k in range(len(path)):
                id_arr.append(path[k])
                phase_arr.append('train')
                input_arr.append(np.array(np.squeeze(inputs[k]).cpu().detach().numpy()))
                output_arr.append(np.squeeze(output[k]).cpu().detach().numpy())

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        vae.eval()
        for inputs, path in valloader:
            #print("==========================VAL==============")
            with torch.no_grad():
                inputs = Variable(inputs).to(device, dtype=torch.float32)
                output, z, logvar = vae(inputs)
                #target = torch.squeeze(inputs, dim=1).long()
                recon_loss_val, kl_val, loss = vae_loss(inputs, output,
                                        z, logvar, criterion,
                                        kl_weight=config.kl)
                #output = torch.argmax(output, dim=1)

                val_loss += loss.cpu().numpy()
                val_steps += 1
        valid_loss = val_loss / val_steps
        images = [inputs[0][0][20][:][:],\
                  output[0][0][20][:][:]]
        writer.add_scalar('Loss/val', valid_loss, epoch)
        writer.add_image('inputs VAL', images[0].unsqueeze(0), epoch)
        writer.add_image('output VAL', images[1].unsqueeze(0), epoch)

        writer.close()
        print("[%d] validation loss: %.3f" % (epoch + 1, valid_loss))
        list_loss_val.append(valid_loss)

        early_stopping(valid_loss, vae)

        """ Saving of reconstructions for visualization in Anatomist software """
        if early_stopping.early_stop or epoch == nb_epoch-1:
            for k in range(len(path)):
                id_arr.append(path[k])
                phase_arr.append('val')
                input_arr.append(np.array(np.squeeze(inputs[k]).cpu().detach().numpy()))
                output_arr.append(np.squeeze(output[k]).cpu().detach().numpy())
            break
    for key, array in {'input': input_arr, 'output' : output_arr,
                           'phase': phase_arr, 'id': id_arr}.items():
        np.save(config.save_dir+key, np.array([array]))
        #np.save(f"/neurospin/dico/lguillon/miccai_22/gridsearch_sub/n_{curr_config['n']}_kl_{curr_config['kl']}/"+key, np.array([array]))

    plot_loss(list_loss_train[1:], config.save_dir+'tot_train_')
    plot_loss(list_loss_val[1:], config.save_dir+'tot_val_')
    #plot_loss(list_loss_train[1:], f"/neurospin/dico/lguillon/miccai_22/gridsearch_sub/n_{curr_config['n']}_kl_{curr_config['kl']}/"+'tot_train_')
    #plot_loss(list_loss_val[1:], f"/neurospin/dico/lguillon/miccai_22/gridsearch_sub/n_{curr_config['n']}_kl_{curr_config['kl']}/"+'tot_val_')
    final_loss_val = list_loss_val[-1:]

    """Saving of trained model"""
    torch.save((vae.state_dict(), optimizer.state_dict()),
                config.save_dir + 'vae.pt')

    print("Finished Training")
    return vae, final_loss_val
