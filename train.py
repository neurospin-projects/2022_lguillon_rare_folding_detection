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
from torchsummary import summary

from model import *
from deep_folding.utils.pytorchtools import EarlyStopping


def train_model(config, trainloader, valloader, root_dir=None):
    """ Trains an inpainting AE for a given hyperparameter configuration
    Args:
        config: instance of class Config
        trainloader: torch loader of training data
        valloader: torch loader of validation data
        root_dir: str, directory where to save model

    Returns:
        model: trained model
        final_loss_val
    """
    torch.manual_seed(0)
    lr = config.lr
    model = InpaintAE(config.in_shape, config.n, depth=3)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    model.to(device)
    summary(model, config.in_shape)

    weights = [1, 2]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    nb_epoch = config.nb_epoch
    early_stopping = EarlyStopping(patience=12, verbose=True, root_dir=root_dir)

    list_loss_train, list_loss_val = [], []

    # arrays enabling to see model reconstructions
    id_arr, phase_arr, input_arr, output_arr = [], [], [], []

    for epoch in range(config.nb_epoch):
        running_loss = 0.0
        epoch_steps = 0
        for (inputs, path), target in trainloader:
            #print(path)
            optimizer.zero_grad()

            inputs = Variable(inputs).to(device, dtype=torch.float32)
            target = torch.squeeze(target, dim=1).long()
            out, z = model(inputs)
            #print('out', out.shape)
            #print('target', target.shape)
            loss = inpaint_loss(out, target, criterion)
            output = torch.argmax(out, dim=1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_steps += 1
        print("[%d] loss: %.3f" % (epoch + 1,
                                        running_loss / epoch_steps))
        list_loss_train.append(running_loss / epoch_steps)
        running_loss = 0.0

        """ Saving of reconstructions for visualization in Anatomist software """
        """if epoch == nb_epoch-1:
            for k in range(len(path)):
                id_arr.append(path[k])
                phase_arr.append('train')
                input_arr.append(np.array(np.squeeze(inputs[k]).cpu().detach().numpy()))
                output_arr.append(np.squeeze(output[k]).cpu().detach().numpy())
        """
        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        model.eval()
        for (inputs, path), target in valloader:
            with torch.no_grad():
                inputs = Variable(inputs).to(device, dtype=torch.float32)
                out, z = model(inputs)
                target = torch.squeeze(target, dim=1).long()
                loss = inpaint_loss(out, target, criterion)
                output = torch.argmax(out, dim=1)

                val_loss += loss.cpu().numpy()
                val_steps += 1
        valid_loss = val_loss / val_steps
        print("[%d] validation loss: %.3f" % (epoch + 1, valid_loss))
        list_loss_val.append(valid_loss)

        early_stopping(valid_loss, model)

        """ Saving of reconstructions for visualization in Anatomist software """
        """if early_stopping.early_stop or epoch == nb_epoch-1:
            for k in range(len(path)):
                id_arr.append(path[k])
                phase_arr.append('val')
                input_arr.append(np.array(np.squeeze(inputs[k]).cpu().detach().numpy()))
                output_arr.append(np.squeeze(output[k]).cpu().detach().numpy())
            break
    for key, array in {'input': input_arr, 'output' : output_arr,
                           'phase': phase_arr, 'id': id_arr}.items():
        np.save(config.save_dir+key, np.array([array]))

    plot_loss(list_loss_train[1:], config.save_dir+'tot_train_')
    plot_loss(list_loss_val[1:], config.save_dir+'tot_val_')
    final_loss_val = list_loss_val[-1:]"""

    """Saving of trained model"""
    """torch.save((model.state_dict(), optimizer.state_dict()),
                config.save_dir + 'model.pt')"""

    print("Finished Training")
    return model
    #return model, final_loss_val
