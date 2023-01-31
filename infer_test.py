import os
import sys
import json
import re
import pandas as pd
from tqdm import tqdm
from soma import aims
import random
import dico_toolbox as dtx

# p = os.path.abspath('../')
# if p not in sys.path:
#     sys.path.append(p)
#
#
# q = os.path.abspath('../../')
# if q not in sys.path:
#     sys.path.append(q)
#
# q = os.path.abspath('../../../')
# if q not in sys.path:
#     sys.path.append(q)

from vae import *
from inference import InpaintDatasetTest


if torch.cuda.is_available():
    device = "cuda:0"


print(1)
model_dir = '/neurospin/dico/lguillon/inpainting/v2/baseline_300/checkpoint.pt'
model = VAE((1, 80, 80, 96), 100, depth=3)
model.load_state_dict(torch.load(model_dir))
model = model.to(device)


subject_dir = "/neurospin/dico/lguillon/distmap/data/"
data_dir = '/neurospin/dico/data/deep_folding/current/datasets/' \
                        'hcp/crops/1mm/SC/no_mask/'

print(2)

subject_dir = "/neurospin/dico/lguillon/distmap/data/"
data_dir = '/neurospin/dico/data/deep_folding/current/datasets/' \
                        'hcp/crops/1mm/SC/no_mask/'

# test_list = pd.read_csv(os.path.join(subject_dir, "test_list.csv"))
#
# filenames = np.load(os.path.join(data_dir,"Rfoldlabels", "sub_id.npy"))
# distmaps = np.load(os.path.join(data_dir, 'Rskel_distmaps_junction', "Rskel_distmaps.npy"),
#                    mmap_mode='r')
#
# skeletons = np.load(os.path.join(data_dir, "Rskeletons_junction",
#                                 "Rskeleton.npy"),
#                    mmap_mode='r')
#
# foldlabels = np.load(os.path.join('/neurospin/dico/data/deep_folding/current/datasets/hcp/foldlabels/raw/junction/crops/',
#                                     "Rlabels.npy"),
#                    mmap_mode='r')
#
# print(distmaps.shape, filenames.shape)
#
# # Selection of test set only
# sorter = np.argsort(filenames)
# filenames_idx = sorter[np.searchsorted(filenames, np.array(test_list['subjects']), sorter=sorter)]
# filenames = filenames[filenames_idx]
# distmaps = distmaps[filenames_idx]
# skeletons = skeletons[filenames_idx]
# foldlabels = foldlabels[filenames_idx]
#
# print(distmaps.shape)
#
# subset = InpaintDatasetTest(foldlabels=foldlabels,
#                             skeletons=skeletons,
#                             distmaps=distmaps,
#                             filenames=filenames,
#                             data_transforms=False)
# testloader = torch.utils.data.DataLoader(
#                subset,
#                batch_size=1,
#                num_workers=1,
#                shuffle=False)


""" SC INT """
scint_list = ['111009', '138231', '510225', '199251', '159946', '140319', '212419']
scint_arr = np.array(['111009', '138231', '510225', '199251', '159946', '140319', '212419'])

filenames_scint = np.load(os.path.join(data_dir, "Rfoldlabels", "sub_id.npy"))
distmaps_scint = np.load(os.path.join(data_dir, 'Rskel_distmaps_junction', "Rskel_distmaps.npy"),
                   mmap_mode='r')

skeletons_scint = np.load(os.path.join(data_dir, "Rskeletons_junction",
                                "Rskeleton.npy"),
                   mmap_mode='r')

foldlabels_scint = np.load(os.path.join('/neurospin/dico/data/deep_folding/current/datasets/hcp/foldlabels/raw/junction/crops/',
                                    "Rlabels.npy"),
                   mmap_mode='r')

ids = np.frompyfunc(lambda x: np.isin(x, scint_arr), 1, 1)(filenames_scint)
idxs = [i for i, curr in enumerate(ids) if curr.any()]

filenames_scint = filenames_scint[idxs]
distmaps_scint = distmaps_scint[idxs]
skeletons_scint = skeletons_scint[idxs]
foldlabels_scint = foldlabels_scint[idxs]

print(distmaps_scint.shape, filenames_scint.shape)

subset_scint = InpaintDatasetTest(foldlabels=foldlabels_scint,
                                   skeletons=skeletons_scint,
                                   distmaps=distmaps_scint,
                                   filenames=filenames_scint,
                                   data_transforms=False)
scint_loader = torch.utils.data.DataLoader(
               subset_scint,
               batch_size=1,
               num_workers=1,
               shuffle=False)

#dico_set_loaders = {'hcp': testloader, 'scint': scint_loader}
dico_set_loaders = {'scint': scint_loader}

# root_dir = '/neurospin/dico/lguillon/inpainting/analyses_gridsearch/'
#
# criterion = nn.MSELoss(reduction='sum')
# tester_hcp = InpaintModelTester(model=model, dico_set_loaders=dico_set_loaders,
#                      loss_func=criterion, kl_weight=2,
#                      n_latent=100, depth=3)
#
# results_hcp = tester_hcp.test()
#
# print(results_hcp['scint'])
# #encoded_hcp = {loader_name:[results_hcp[loader_name][k][0] for k in results_hcp[loader_name].keys()] for loader_name in dico_set_loaders.keys()}
# losses_hcp = {loader_name:[results_hcp[loader_name][k][0] for k in results_hcp[loader_name].keys()] for loader_name in dico_set_loaders.keys()}
# recon_hcp = {loader_name:[results_hcp[loader_name][k][1] for k in results_hcp[loader_name].keys()] for loader_name in dico_set_loaders.keys()}
#
#
# print(recon_hcp)

with torch.no_grad():
    for distmap_masked_dict, distmap, path in scint_loader:
        out_z = []
        print(distmap_masked_dict.keys())
        print(path)
        for ss_size, ss in distmap_masked_dict.items():
            inputs = Variable(ss).to(device, dtype=torch.float32)
            distmap = Variable(distmap).to(device, dtype=torch.float32)
            output, z, logvar = model(inputs)
            #target = torch.squeeze(inputs, dim=1).long()
            #recon_loss_val, kl_val, loss_val = vae_loss(distmap, output, z, logvar, self.loss_func,
            #                kl_weight=self.kl_weight)
            # out_z = np.array(np.squeeze(z).cpu().detach().numpy())
            print('ici')
            #z = torch.from_numpy(z[0]).to(device, dtype=torch.float32)
            z = torch.unsqueeze(z, dim=0)
            out = model.decode(z)
            print('la')
            #output = torch.argmax(out, dim=1)
            out = np.array(np.squeeze(out).cpu().detach().numpy())
            inputs = np.array(np.squeeze(inputs).cpu().detach().numpy())
            distmap = np.array(np.squeeze(distmap).cpu().detach().numpy())
            print(out.shape)
            out[out<0.5]=0
            out[out>=0.5]=1
            inputs[inputs<0.4]=0
            inputs[inputs>=0.4]=1
            distmap[distmap<0.4]=0
            distmap[distmap>=0.4]=1
            #aims.write(aims.Volume(out), "/tmp_out.nii.gz")
            aims.write(dtx.convert.bucket_to_mesh(dtx.convert.volume_to_bucket_numpy(out)),
                f"/tmp/mesh_{path}_{ss_size}.mesh")
            aims.write(dtx.convert.volume_to_bucketMap_aims(out),
                f"/tmp/mesh_{path}_{ss_size}.bck")
            aims.write(dtx.convert.bucket_to_mesh(dtx.convert.volume_to_bucket_numpy(inputs)),
                f"/tmp/input_{path}_{ss_size}.mesh")
            aims.write(dtx.convert.bucket_to_mesh(dtx.convert.volume_to_bucket_numpy(distmap)),
                f"/tmp/target_{path}_{ss_size}.mesh")

# aims.write(dtx.convert.bucket_to_mesh(dtx.convert.volume_to_bucket_numpy(error)),
#         f"{tgt_dir}{sub}_error_missing.mesh")
# aims.write(dtx.convert.bucket_to_mesh(dtx.convert.volume_to_bucket_numpy(error2)),
#            f"{tgt_dir}{sub}_error_adding.mesh")
# aims.write(dtx.convert.bucket_to_mesh(dtx.convert.volume_to_bucket_numpy(input_ctrl)),
#         f"{tgt_dir}{sub}_input.mesh")
# aims.write(dtx.convert.bucket_to_mesh(dtx.convert.volume_to_bucket_numpy(out)),
#            f"{tgt_dir}{sub}_recon.mesh")

# results = {k:{} for k in dico_set_loaders.keys()}
# out_z = []
#
# for loader_name, loader in dico_set_loaders.items():
#     model.eval()
#     with torch.no_grad():
#         for distmap_masked_dict, distmap, path in loader:
#             print(path)
#             for ss_size, ss in distmap_masked_dict.items():
#                 print('here', ss_size)
#                 #for ss, distmap, path in loader:
#                 inputs = Variable(ss).to(device, dtype=torch.float32)
#                 distmap = Variable(distmap).to(device, dtype=torch.float32)
#                 output, z, logvar = model(inputs)
#                 #target = torch.squeeze(inputs, dim=1).long()
#                 recon_loss_val, kl_val, loss_val = vae_loss(distmap, output, z, logvar, self.loss_func,
#                                  kl_weight=self.kl_weight)
#
#                 for k in range(len(path)):
#                     out_z = np.array(np.squeeze(z[k]).cpu().detach().numpy())
#                     #results[loader_name][path[k]] = loss_val, out_z, recon_loss_val
#                     results[loader_name][path[k]] = loss_val, out_z, recon_loss_val, inputs, output
