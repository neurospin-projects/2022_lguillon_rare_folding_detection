""" The aim of this script is to display saved model outputs thanks to Anatomist.
Model outputs are stored as numpy arrays.
"""

import anatomist.api as anatomist
from soma import aims
import numpy as np
import dico_toolbox as dtx


def array_to_ana(ana_a, img, sub_id, phase, status):
    """
    Transforms output tensor into volume readable by Anatomist and defines
    name of the volume displayed in Anatomist window.
    Returns volume displayable by Anatomist
    """
    #vol_img = aims.Volume(img)
    vol_img = img
    a_vol_img = ana_a.toAObject(vol_img)
    vol_img.header()['voxel_size'] = [1, 1, 1]
    a_vol_img.setName(status+'_'+ str(sub_id)+'_'+str(phase)) # display name
    a_vol_img.setChanged()
    a_vol_img.notifyObservers()

    return vol_img, a_vol_img


def main():
    """
    In the Anatomist window, for each model output, corresponding input will
    also be displayed at its left side.
    Number of columns and view (Sagittal, coronal, frontal) can be specified.
    (It's better to choose an even number for number of columns to display)
    """
    root_dir = "/neurospin/dico/lguillon/distmap/"
    root_dir = '/neurospin/dico/lguillon/2022_lguillon_foldInpainting/skel_comparison/'

    a = anatomist.Anatomist()
    block = a.AWindowsBlock(a, 12)  # Parameter 6 corresponds to the number of columns displayed. Can be changed.

    #input_arr = np.load(root_dir+'input.npy').astype('float32') # Input
    output_arr = np.load(root_dir+"out.npy").astype('float32') # Input
    #id_arr = np.load(root_dir+'id.npy') # Subject id

    #for k in range(len(output_arr)):
    for k in range(1):
        # img = input_arr[k]
        output = output_arr.astype(float)
        output = dtx.convert.volume_to_bucketMap_aims(output, voxel_size=(1,1,1))
        # sub_id = id_arr[k]
        #for img, entry in [(input, 'input'), (output, 'output')]:
        for img, entry in [(output, 'output')]:
            sub_id = 'ave'
            globals()['block%s%s' % (sub_id, entry)] = a.createWindow('Sagittal', block=block)

            globals()['img%s%s' % (sub_id, entry)], globals()['a_img%s%s' % (sub_id, entry)] = array_to_ana(a, img, sub_id, phase='', status=entry)

            globals()['block%s%s' % (sub_id, entry)].addObjects(globals()['a_img%s%s' % (sub_id, entry)])



if __name__ == '__main__':
    main()
    from soma.qt_gui.qt_backend import Qt
    Qt.qApp.exec_()
