import os
import sys
import json
import re
import pandas as pd
from tqdm import tqdm
from soma import aims

p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)

q = os.path.abspath('../../')
if q not in sys.path:
    sys.path.append(q)

from vae import *
from preprocess import SkeletonDataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import dico_toolbox as dtx
import umap
from scipy.spatial import distance
from scipy.stats import mannwhitneyu

from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def infer(n, kl):
    if torch.cuda.is_available():
        device = "cuda:0"

    #model_dir = f"/neurospin/dico/lguillon/distmap/gridsearch_lr5e-4/n_{n}_kl_{kl}/checkpoint.pt"
    model_dir = f"/neurospin/dico/lguillon/distmap/rotation_-3_3/checkpoint.pt"
    model = VAE((1, 80, 80, 96), n, depth=3)
    model.load_state_dict(torch.load(model_dir))
    model = model.to(device)
    print("model loaded")

    """ HCP TEST SET """
    subject_dir = "/neurospin/dico/lguillon/distmap/data/"
    data_dir = "/neurospin/dico/lguillon/distmap/data/"
    test_list = pd.read_csv(os.path.join(subject_dir, "test_list.csv"))
    distmaps = np.load(os.path.join(data_dir, "test_distmap.npy"),
                       mmap_mode='r')
    filenames = np.load(os.path.join(data_dir, "test_sub_id.npy"))
    print(distmaps.shape, filenames.shape)

    subset = SkeletonDataset(dataframe=distmaps,
                             filenames=filenames,
                             data_transforms=False)
    testloader = torch.utils.data.DataLoader(
                   subset,
                   batch_size=1,
                   num_workers=1,
                   shuffle=False)

    """ SC INT """
    scint_list = ['111009', '138231', '510225', '199251', '159946', '140319', '212419']
    scint_arr = np.array(['111009', '138231', '510225', '199251', '159946', '140319', '212419'])
    data_dir = "/neurospin/dico/data/deep_folding/current/datasets/hcp/crops/1mm/SC/no_mask/Rdistmaps"

    distmaps_scint = np.load(os.path.join(data_dir, "distmap_1mm.npy"), mmap_mode='r')
    filenames_scint = np.load(os.path.join(data_dir, "sub_id.npy"))

    ids = np.frompyfunc(lambda x: np.isin(x, scint_arr), 1, 1)(filenames_scint)
    idxs = [i for i, curr in enumerate(ids) if curr.any()]

    filenames_scint = filenames_scint[idxs]
    distmaps_scint = distmaps_scint[idxs]

    print(distmaps_scint.shape, filenames_scint.shape)

    subset_scint = SkeletonDataset(dataframe=distmaps_scint,
                             filenames=filenames_scint,
                             data_transforms=False)
    scint_loader = torch.utils.data.DataLoader(
                   subset_scint,
                   batch_size=1,
                   num_workers=1,
                   shuffle=False)

    dico_set_loaders_hcp = {'hcp': testloader, 'scint': scint_loader}

    root_dir = '/neurospin/dico/lguillon/distmap/analyses_gridsearch/'
    root_dir = '/neurospin/dico/lguillon/distmap/rotation_-3_3'

    criterion = nn.MSELoss(reduction='sum')
    tester_hcp = ModelTester(model=model, dico_set_loaders=dico_set_loaders_hcp,
                         loss_func=criterion, kl_weight=kl,
                         n_latent=n, depth=3)

    results_hcp = tester_hcp.test()
    encoded_hcp = {loader_name:[results_hcp[loader_name][k][1] for k in results_hcp[loader_name].keys()] for loader_name in dico_set_loaders_hcp.keys()}
    losses_hcp = {loader_name:[int(results_hcp[loader_name][k][0].cpu().detach().numpy()) for k in results_hcp[loader_name].keys()] for loader_name in dico_set_loaders_hcp.keys()}
    recon_hcp = {loader_name:[int(results_hcp[loader_name][k][2].cpu().detach().numpy()) for k in results_hcp[loader_name].keys()] for loader_name in dico_set_loaders_hcp.keys()}
    input_hcp = {loader_name:[results_hcp[loader_name][k][3].cpu().detach().numpy() for k in results_hcp[loader_name].keys()] for loader_name in dico_set_loaders_hcp.keys()}

    df_encoded_hcp = pd.DataFrame()
    df_encoded_hcp['latent'] = encoded_hcp['hcp'] + encoded_hcp['scint']
    df_encoded_hcp['loss'] = losses_hcp['hcp'] + losses_hcp['scint']
    df_encoded_hcp['recon'] = recon_hcp['hcp'] + recon_hcp['scint']
    df_encoded_hcp['input'] = input_hcp['hcp'] + input_hcp['scint']
    df_encoded_hcp['Group'] = ['hcp' for k in range(len(filenames))] + ['scint' for k in range(len(filenames_scint))]
    df_encoded_hcp['sub'] = list(filenames) + list(filenames_scint)
    print(df_encoded_hcp.head())
    df_encoded_hcp.to_pickle(f"/neurospin/dico/lguillon/distmap/rotation_-3_3/encoded_hcp.pkl")

    """ One-handed subjects """
    print("One handed dataset")
    labels_oh = pd.read_csv('/neurospin/dico/lguillon/ohbm_22/one_handed_labels.csv')
    ctrl = labels_oh[labels_oh['Dominant hand']=='R']
    one_handed = labels_oh[labels_oh['Group']!='CTR']
    one_handed = labels_oh[labels_oh['Amp. Side']=='L']

    amputee = one_handed[one_handed['Group']=='AMP']
    cong = one_handed[one_handed['Group']=='CONG']

    print(len(ctrl))

    data_dir = "/neurospin/dico/data/deep_folding/current/datasets/one_handed/crops/SC/no_mask/Rdistmaps/"

    distmaps_ctrl = np.load(os.path.join(data_dir, "distmap_1mm.npy"), mmap_mode='r')
    filenames_ctrl = np.load(os.path.join(data_dir, "sub_id.npy"))

    distmaps_amputee = np.load(os.path.join(data_dir, "distmap_1mm.npy"), mmap_mode='r')
    filenames_amputee = np.load(os.path.join(data_dir, "sub_id.npy"))

    distmaps_cong = np.load(os.path.join(data_dir, "distmap_1mm.npy"), mmap_mode='r')
    filenames_cong = np.load(os.path.join(data_dir, "sub_id.npy"))

    ctrl_list = np.array(list(ctrl.SubjID))
    sorter = np.argsort(filenames_ctrl)
    filenames_idx = sorter[np.searchsorted(filenames_ctrl, ctrl_list, sorter=sorter)]
    filenames_ctrl = filenames_ctrl[filenames_idx]
    distmaps_ctrl = distmaps_ctrl[filenames_idx]

    amputee_list = np.array(list(amputee.SubjID))
    sorter = np.argsort(filenames_amputee)
    filenames_idx = sorter[np.searchsorted(filenames_amputee, amputee_list, sorter=sorter)]
    filenames_amputee = filenames_amputee[filenames_idx]
    distmaps_amputee = distmaps_amputee[filenames_idx]

    cong_list = np.array(list(cong.SubjID))
    sorter = np.argsort(filenames_cong)
    filenames_idx = sorter[np.searchsorted(filenames_cong, cong_list, sorter=sorter)]
    filenames_cong = filenames_cong[filenames_idx]
    distmaps_cong = distmaps_cong[filenames_idx]

    subset_ctrl = SkeletonDataset(dataframe=distmaps_ctrl,
                         filenames=filenames_ctrl,
                         data_transforms=False)
    ctrl_loader = torch.utils.data.DataLoader(
                   subset_ctrl,
                   batch_size=1,
                   num_workers=1,
                   shuffle=False)
    subset_amputee = SkeletonDataset(dataframe=distmaps_amputee,
                             filenames=filenames_amputee,
                             data_transforms=False)

    amputee_loader = torch.utils.data.DataLoader(
                   subset_amputee,
                   batch_size=1,
                   num_workers=1,
                   shuffle=False)
    subset_cong = SkeletonDataset(dataframe=distmaps_cong,
                             filenames=filenames_cong,
                             data_transforms=False)
    cong_loader = torch.utils.data.DataLoader(
                   subset_cong,
                   batch_size=1,
                   num_workers=1,
                   shuffle=False)

    dico_set_loaders_oh = {'ctrl': ctrl_loader, 'amputee': amputee_loader, 'congenital': cong_loader}
    tester_oh = ModelTester(model=model, dico_set_loaders=dico_set_loaders_oh,
                     loss_func=criterion, kl_weight=kl,
                     n_latent=n, depth=3)
    results_oh = tester_oh.test()
    losses_oh = {loader_name:[int(results_oh[loader_name][k][0].cpu().detach().numpy()) for k in results_oh[loader_name].keys()] for loader_name in dico_set_loaders_oh.keys()}
    encoded_oh = {loader_name:[results_oh[loader_name][k][1] for k in results_oh[loader_name].keys()] for loader_name in dico_set_loaders_oh.keys()}
    recon_oh = {loader_name:[int(results_oh[loader_name][k][2].cpu().detach().numpy()) for k in results_oh[loader_name].keys()] for loader_name in dico_set_loaders_oh.keys()}
    input_oh = {loader_name:[results_oh[loader_name][k][3].cpu().detach().numpy() for k in results_oh[loader_name].keys()] for loader_name in dico_set_loaders_oh.keys()}

    df_encoded_oh = pd.DataFrame()
    df_encoded_oh['latent'] = encoded_oh['ctrl'] + encoded_oh['amputee'] + encoded_oh['congenital']
    df_encoded_oh['loss'] = losses_oh['ctrl'] + losses_oh['amputee'] + losses_oh['congenital']
    df_encoded_oh['recon'] = recon_oh['ctrl'] + recon_oh['amputee'] + recon_oh['congenital']
    df_encoded_oh['input'] = input_oh['ctrl'] + input_oh['amputee'] + input_oh['congenital']
    df_encoded_oh['Group'] = ['ctrl' for k in range(len(losses_oh['ctrl']))] + ['amputee' for k in range(len(losses_oh['amputee']))] +['congenital'for k in range(len(losses_oh['congenital']))]
    df_encoded_oh['sub'] = list(filenames_ctrl) + list(filenames_amputee) + list(filenames_cong)
    df_encoded_oh.to_pickle(f"/neurospin/dico/lguillon/distmap/rotation_-3_3/encoded_oh.pkl")

    """ EU AIMS """
    labels_aims = pd.read_csv('/neurospin/dico/lguillon/aims_detection/list_subjects.csv')
    ctrl = labels_aims[labels_aims['group']==1]
    asd = labels_aims[labels_aims['group']==2]
    id_ctrl = labels_aims[labels_aims['group']==3]

    data_dir = "/neurospin/dico/data/deep_folding/current/datasets/euaims/crops/SC/no_mask/Rdistmaps/"
    filenames = np.load(os.path.join(data_dir, "sub_id.npy"))
    filenames = np.array([re.search('\d{12}', filenames[k]).group(0) for k in range(len(filenames))])

    distmaps_ctrl = np.load(os.path.join(data_dir, "distmap_1mm.npy"), mmap_mode='r')
    distmaps_asd = np.load(os.path.join(data_dir, "distmap_1mm.npy"), mmap_mode='r')
    distmaps_id_ctrl = np.load(os.path.join(data_dir, "distmap_1mm.npy"), mmap_mode='r')

    ctrl_list = [str(sub) for sub in ctrl.ID if str(sub) in filenames]
    ctrl_list = np.array(ctrl_list)
    sorter = np.argsort(filenames)
    filenames_idx = sorter[np.searchsorted(filenames, ctrl_list, sorter=sorter)]
    filenames_ctrl = filenames[filenames_idx]
    filenames_ctrl = np.array([re.search('\d{12}', filenames_ctrl[k]).group(0) for k in range(len(filenames_ctrl))])
    distmaps_ctrl = distmaps_ctrl[filenames_idx]

    asd_list = [str(sub) for sub in asd.ID if str(sub) in filenames]
    asd_list = np.array(asd_list)
    filenames_idx = sorter[np.searchsorted(filenames, asd_list, sorter=sorter)]
    filenames_asd = filenames[filenames_idx]
    distmaps_asd = distmaps_asd[filenames_idx]

    id_ctrl_list = np.array(list(id_ctrl.ID))
    sorter = np.argsort(filenames)
    filenames_idx = sorter[np.searchsorted(filenames, id_ctrl_list, sorter=sorter)]
    filenames_id_ctrl = filenames[filenames_idx]
    distmaps_id_ctrl = distmaps_id_ctrl[filenames_idx]

    subset_ctrl = SkeletonDataset(dataframe=distmaps_ctrl,
                         filenames=filenames_ctrl,
                         data_transforms=False)

    ctrl_loader = torch.utils.data.DataLoader(
                   subset_ctrl,
                   batch_size=1,
                   num_workers=8,
                   shuffle=False)

    subset_asd = SkeletonDataset(dataframe=distmaps_asd,
                             filenames=filenames_asd,
                             data_transforms=False)

    asd_loader = torch.utils.data.DataLoader(
                   subset_asd,
                   batch_size=1,
                   num_workers=1,
                   shuffle=False)
    subset_id_ctrl = SkeletonDataset(dataframe=distmaps_id_ctrl,
                             filenames=filenames_id_ctrl,
                             data_transforms=False)

    id_ctrl_loader = torch.utils.data.DataLoader(
                   subset_id_ctrl,
                   batch_size=1,
                   num_workers=1,
                   shuffle=False)

    dico_set_loaders_aims = {'ctrl': ctrl_loader,
                             'asd': asd_loader,
                             'id_ctrl': id_ctrl_loader}

    tester_aims = ModelTester(model=model, dico_set_loaders=dico_set_loaders_aims,
                     loss_func=criterion, kl_weight=kl,
                     n_latent=n, depth=3)

    results_aims = tester_aims.test()

    losses_aims = {loader_name:[int(results_aims[loader_name][k][0].cpu().detach().numpy()) for k in results_aims[loader_name].keys()] for loader_name in dico_set_loaders_aims.keys()}
    encoded_aims = {loader_name:[results_aims[loader_name][k][1] for k in results_aims[loader_name].keys()] for loader_name in dico_set_loaders_aims.keys()}
    recon_aims = {loader_name:[int(results_aims[loader_name][k][2].cpu().detach().numpy()) for k in results_aims[loader_name].keys()] for loader_name in dico_set_loaders_aims.keys()}
    input_aims = {loader_name:[results_aims[loader_name][k][3].cpu().detach().numpy() for k in results_aims[loader_name].keys()] for loader_name in dico_set_loaders_aims.keys()}
    df_encoded_aims = pd.DataFrame()
    df_encoded_aims['latent'] = encoded_aims['ctrl'] + encoded_aims['asd'] + encoded_aims['id_ctrl']
    df_encoded_aims['loss'] = losses_aims['ctrl'] + losses_aims['asd'] + losses_aims['id_ctrl']
    df_encoded_aims['recon'] = recon_aims['ctrl'] + recon_aims['asd'] + recon_aims['id_ctrl']
    df_encoded_aims['input'] = input_aims['ctrl'] + input_aims['asd'] + input_aims['id_ctrl']
    df_encoded_aims['Group'] = ['ctrl' for k in range(len(losses_aims['ctrl']))] + ['asd' for k in range(len(losses_aims['asd']))] + ['id_ctrl' for k in range(len(losses_aims['id_ctrl']))]
    df_encoded_aims['sub'] = list(filenames_ctrl) + list(filenames_asd) + list(filenames_id_ctrl)
    df_encoded_aims.to_pickle(f"/neurospin/dico/lguillon/distmap/rotation_-3_3/encoded_aims.pkl")


def main():
    infer(75, 2)

if __name__ == '__main__':
    main()
