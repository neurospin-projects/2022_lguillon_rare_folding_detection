#!/usr/bin/env python
# coding: utf-8

# # Benchmarks analysis

# This notebook aims to assess our model's performances on two synthetic benchkmarks of abnormalities:
# - deletion benchmark: simple surfaces of various sizes have been randomly deleted
# - asymmetry benchmark: equivalent crop but on left hemisphere and then flipped

# In[1]:


import os
import sys
import json
import re
import pandas as pd
from tqdm import tqdm
from soma import aims
import random

# p = os.path.abspath('../')
# if p not in sys.path:
#     sys.path.append(p)
#
# q = os.path.abspath('../../')
# if q not in sys.path:
#     sys.path.append(q)

from vae import *
from preprocess import SkeletonDataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import dico_toolbox as dtx
import umap
from scipy.spatial import distance
from scipy.interpolate import interp1d
from scipy.stats import mannwhitneyu, ttest_ind, ks_2samp

from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.svm import LinearSVC


# In[2]:


if torch.cuda.is_available():
    device = "cuda:0"


# In[3]:


model_dir = '/neurospin/dico/lguillon/distmap/gridsearch_lr5e-4/n_75_kl_2/checkpoint.pt'
model = VAE((1, 80, 80, 96), 75, depth=3)
model.load_state_dict(torch.load(model_dir))
model = model.to(device)


# In[4]:


data_dir = "/neurospin/dico/lguillon/distmap/analyses_gridsearch/75_2/"
df_encoded_hcp = pd.read_pickle(os.path.join(data_dir, "encoded_hcp.pkl"))


# ## Benchmark Asymmetry

# In[5]:


data_dir = '/neurospin/dico/lguillon/distmap/benchmark/asymmetry/'


# In[6]:


distmaps_asym = np.load(os.path.join(data_dir, "asym_benchmark_1mm.npy"),
                   mmap_mode='r')
filenames_asym = np.load(os.path.join(data_dir, "sub_id.npy"))

subset_asym = SkeletonDataset(dataframe=distmaps_asym,
                         filenames=filenames_asym,
                         data_transforms=False)
loader_asym = torch.utils.data.DataLoader(
               subset_asym,
               batch_size=1,
               num_workers=1,
               shuffle=False)


# In[7]:


dico_set_loaders = {'asymmetry': loader_asym}

root_dir = '/neurospin/dico/lguillon/distmap/benchmark/results/'

criterion = nn.MSELoss(reduction='sum')
tester_asym = ModelTester(model=model, dico_set_loaders=dico_set_loaders,
                     loss_func=criterion, kl_weight=2,
                     n_latent=75, depth=3)

results_asym = tester_asym.test()
encoded_asym = {loader_name:[results_asym[loader_name][k][1] for k in results_asym[loader_name].keys()] for loader_name in dico_set_loaders.keys()}
losses_asym = {loader_name:[int(results_asym[loader_name][k][0].cpu().detach().numpy()) for k in results_asym[loader_name].keys()] for loader_name in dico_set_loaders.keys()}
recon_asym = {loader_name:[int(results_asym[loader_name][k][2].cpu().detach().numpy()) for k in results_asym[loader_name].keys()] for loader_name in dico_set_loaders.keys()}
input_asym = {loader_name:[results_asym[loader_name][k][3].cpu().detach().numpy() for k in results_asym[loader_name].keys()] for loader_name in dico_set_loaders.keys()}


# In[8]:


df_encoded_asym = pd.DataFrame()
df_encoded_asym['latent'] = encoded_asym['asymmetry']
df_encoded_asym['loss'] = losses_asym['asymmetry']
df_encoded_asym['recon'] = recon_asym['asymmetry']
df_encoded_asym['input'] = input_asym['asymmetry']
df_encoded_asym['sub'] = list(filenames_asym)


# In[9]:


list_encoded_asym = random.sample(list(df_encoded_asym['sub']), 100)
df_encoded_asym_X = df_encoded_asym[df_encoded_asym['sub'].astype(int).isin(list_encoded_asym)]

list_ctrl = list(set(list(df_encoded_hcp['sub'][:200])) - set(list(df_encoded_asym_X['sub'].astype(int))))
df_encoded_hcp_X = df_encoded_hcp[df_encoded_hcp['sub'].isin(list_ctrl)]

X_asym = np.array(list(df_encoded_asym_X['latent']))
X_hcp = np.array(list(df_encoded_hcp_X['latent']))
X_all = np.array(list(df_encoded_hcp_X['latent']) + list(df_encoded_asym_X['latent']))

labels_asym = np.array(list(df_encoded_hcp_X['Group']) + ['asymmetry' for k in range(len(df_encoded_asym_X))])
# reducer = umap.UMAP(random_state=14)
# embedding_asym = reducer.fit_transform(X_all)
#
#
# # In[10]:
#
#
# arr = embedding_asym
# color_dict = {'hcp':'mediumseagreen', 'asymmetry': 'royalblue'}
#
# fig = plt.figure(figsize=(6,4))
# ax = fig.subplots()
#
# for g in np.unique(labels_asym):
#     ix = np.where(labels_asym == g)
#     x = [arr[ix][k][0] for k in range(len(ix[0]))]
#     y = [arr[ix][k][1] for k in range(len(ix[0]))]
#     g_lab= f"{g}"
#     ax.scatter(x, y, c = color_dict[g], label = g_lab)
#
# plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
#
# plt.xlabel(f'UMAP dimension 1', fontsize=14)
# plt.ylabel(f'UMAP dimension 2', fontsize=14)
# plt.show()
#
#
# # In[237]:
#
#
# enc = [z_ctrl[0]] + [z_ctrl[0] + ((z_asym[0]-z_ctrl[0])/5)*k for k in [1,2,3,4]] + [z_asym[0]]
#
#
# # In[16]:
#
#
# df_encoded_asym_X
#
#
# # In[19]:
#
#
# #z_ctrl = df_encoded_hcp_X[df_encoded_hcp_X['sub']==844961].latent.values
z_asym = df_encoded_asym_X.iloc[0].latent

# list_enc = [z_ctrl[0]] + [z_ctrl[0] + ((z_asym[0]-z_ctrl[0])/5)*k for k in [1,2,3,4]] + [z_asym[0]]

# arr_out = []
# values = []

# for k in range(6):
#     enc = list_enc[k]
#     z = torch.from_numpy(np.array(enc)).to(device, dtype=torch.float32)
#     z = torch.unsqueeze(z, dim=0)
#     out = model.decode(z)
#     output = torch.argmax(out, dim=1)
#     out = np.array(np.squeeze(output).cpu().detach().numpy())
#     arr_out.append(out)
#     values.append(f"step_{k}")


# In[20]:


# z_asym


# In[ ]:


for inputs, path in loader_asym:
    print(path)
    with torch.no_grad():
        inputs = Variable(inputs).to(device, dtype=torch.float32)
        output, z, logvar = model(inputs)
        print('version1', torch.unique(output))
        # z = torch.from_numpy(z_asym).to(device, dtype=torch.float32)
        # z = torch.unsqueeze(z, dim=0)
        out = model.decode(z)
        print('v2',torch.unique(out))
        #output = torch.argmax(out, dim=1)
        out = np.array(np.squeeze(output).cpu().detach().numpy())
        print('v2',np.unique(out))


# In[22]:


z = torch.from_numpy(z_asym).to(device, dtype=torch.float32)
z = torch.unsqueeze(z, dim=0)
out = model.decode(z)
#output = torch.argmax(out, dim=1)
out = np.array(np.squeeze(out).cpu().detach().numpy())


# In[23]:

print('iciiii')
print(np.unique(out))


# In[262]:


tgt_dir = '/neurospin/dico/lguillon/distmap/results_benchmark/'
for k in range(6):
    np.save(f"{tgt_dir}skel_step_{k}", out)
#     aims.write(dtx.convert.bucket_numpy_to_bucketMap_aims(dtx.convert.volume_to_bucket_numpy(arr_out[k]),
#                                                           voxel_size=(1,1,1)),
#                 f"{tgt_dir}interpolation_bucket_step_{k}.bck")
