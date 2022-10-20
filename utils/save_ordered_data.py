"""
Scripts that enables to create a dataframe of numpy arrays from .nii.gz or .nii
images in a specific given order.
It is specially interesting for inpainting task when distmap, foldlabel and
skeleton subjects must be in the same order.
"""

import os
import re
import glob

import numpy as np
import pandas as pd
from soma import aims

from deep_folding.config.logs import set_file_logger

log = set_file_logger(__file__)

def is_file_nii(filename):
    """Tests if file is nii file

    Args:
        filename: string giving file name with full path

    Returns:
        is_file_nii: boolean stating if file is nii file
    """
    is_file_nii = os.path.isfile(filename)\
        and '.nii' in filename \
        and '.minf' not in filename
    return is_file_nii


def quality_checks(csv_file_path, npy_array_file_path, cropped_dir):
    """Checks that the numpy arrays are equal to subject nifti files.

    This is to check that the subjects list in csv file
    match the order set in numpy arrays"""
    arr = np.load(npy_array_file_path, mmap_mode='r')
    subjects = pd.read_csv(csv_file_path)
    log.info(f"subjects.head() = {subjects.head()}")
    for index, row in subjects.iterrows():
        sub = row['Subject']
        subject_file = glob.glob(f"{cropped_dir}/{sub}*.nii.gz")[0]
        vol = aims.read(subject_file)
        arr_ref = np.asarray(vol)
        arr_from_array = arr[index,...]
        if not np.array_equal(arr_ref, arr_from_array):
            raise ValueError(f"For subject = {sub} and index = {index}\n"
                              "arrays don't match")


def save_to_numpy(cropped_dir, tgt_dir=None, file_basename=None):
    """
    Creates a numpy array for each subject.

    Saved these this dataframe to npy format on the target
    directory

    Args:
        cropped_dir: directory containing cropped images
        tgt_dir: directory where to save the numpy array file
        file_basename: final file name = file_basename.npy
    """
    list_sample_id = []
    list_sample_file = []

    list_subjects = np.load('/neurospin/dico/data/deep_folding/current/datasets/hcp/' \
                    'crops/1mm/SC/no_mask/Rfoldlabels/sub_id.npy')
    for sub in list_subjects:
        file_nii = os.path.join(cropped_dir, f"{sub}_cropped_foldlabel.nii.gz")
        if is_file_nii(file_nii):
            aimsvol = aims.read(file_nii)
            sample = np.asarray(aimsvol)
            list_sample_id.append(sub)
            list_sample_file.append(sample)

    # Writes subject ID csv file
    subject_df = pd.DataFrame(list_sample_id, columns=["Subject"])
    subject_df.to_csv(os.path.join(tgt_dir, file_basename+'_subject.csv'),
                      index=False)
    log.info(f"5 first saved subjects are: {subject_df.head()}")

    # Writes subject ID to npy file (per retrocompatibility)
    list_sample_id = np.array(list_sample_id)
    np.save(os.path.join(tgt_dir, 'sub_id.npy'), list_sample_id)

    # Writes volumes as numpy arrays
    list_sample_file = np.array(list_sample_file)
    np.save(os.path.join(tgt_dir, file_basename+'.npy'), list_sample_file)

    # Quality_checks
    log.info("Now performing checks on numpy arrays...")
    quality_checks(
        os.path.join(tgt_dir, file_basename+'_subject.csv'),
        os.path.join(tgt_dir, file_basename+'.npy'),
        cropped_dir)


if __name__ == '__main__':
    # save_to_pickle(
    #     cropped_dir='/neurospin/dico/data/deep_folding/current/crops/SC/mask/sulcus_based/2mm/Rlabels/',
    #     tgt_dir='/neurospin/dico/data/deep_folding/current/crops/SC/mask/sulcus_based/2mm/',
    #     file_basename='Rlabels')
    # save_to_numpy(
    #     cropped_dir='/neurospin/dico/data/deep_folding/current/datasets/hcp/crops/1mm/SC/no_mask/Rskeletons_junction/Rcrops',
    #     tgt_dir='/neurospin/dico/data/deep_folding/current/datasets/hcp/crops/1mm/SC/no_mask/Rskeletons_junction',
    #     file_basename='Rskeleton')
    save_to_numpy(
        cropped_dir='/neurospin/dico/data/deep_folding/current/datasets/hcp/foldlabels/raw/junction/crops/Rlabels',
        tgt_dir='/neurospin/dico/data/deep_folding/current/datasets/hcp/foldlabels/raw/junction/crops',
        file_basename='Rlabels')
    # save_to_numpy(
    #     cropped_dir='/neurospin/dico/data/deep_folding/current/datasets/hcp/crops/1mm/SC/no_mask/Rskel_distmaps_junction/R',
    #     tgt_dir='/neurospin/dico/data/deep_folding/current/datasets/hcp/crops/1mm/SC/no_mask/Rskel_distmaps_junction/',
    #     file_basename='Rskel_distmaps')
