""" The aim of this script is to display saved model outputs thanks to Anatomist.
Model outputs are stored as numpy arrays.
"""

import anatomist.api as anatomist
from soma import aims
import dico_toolbox as dtx
import numpy as np


def array_to_ana(ana_a, img, phase, status):
    """
    Transforms output tensor into volume readable by Anatomist and defines
    name of the volume displayed in Anatomist window.
    Returns volume displayable by Anatomist
    """
    #vol_img = aims.Volume(img)
    vol_img = img
    a_vol_img = ana_a.toAObject(vol_img)
    vol_img.header()['voxel_size'] = [1, 1, 1]
    a_vol_img.setName(status+'_'+'_'+str(phase)) # display name
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
    root_dir = "/neurospin/dico/lguillon/distmap/analyses_gridsearch/75_2/through_latent_space/"
    buckets = True

    a = anatomist.Anatomist()
    block = a.AWindowsBlock(a, 12)  # Parameter 6 corresponds to the number of columns displayed. Can be changed.

    #input_arr = np.load(root_dir+'centroid_input.npy').astype('float32') # Input
    output_arr = np.load(root_dir+'centroid_out.npy').astype('float32') # Input
    #error_arr = np.load(root_dir+'centroid_error.npy').astype('float32')  # Subject id
    #print(input_arr.shape)

    output = output_arr.astype(float)
    for k in range(len(output_arr)):
        #input = input_arr[k][0]
        output = output_arr
        #error_map = error_arr[k][0]

        if buckets :
            # input[input>0.5] = 1
            # input[input<=0.5] = 0
            output[output>0.4] = 1
            output[output<=0.4] = 0
            # error_map[error_map>0.7] = 1
            # error_map[error_map<=0.7] = 0
            # print(input.shape, output.shape, error_map.shape)
            # input = dtx.convert.volume_to_bucketMap_aims(input, voxel_size=(1,1,1))
            output = dtx.convert.volume_to_bucketMap_aims(output, voxel_size=(1,1,1))
            # error_map = dtx.convert.volume_to_bucketMap_aims(error_map, voxel_size=(1,1,1))

        #for img, entry in [(input, 'input'), (output, 'output'), (error_map, 'error_map')]:
        for img, entry in [(output, 'output')]:
            globals()['block%s' % (entry)] = a.createWindow('Sagittal', block=block)

            globals()['img%s' % (entry)], globals()['a_img%s' % (entry)] = array_to_ana(a, img, phase='', status=entry)

            globals()['block%s' % (entry)].addObjects(globals()['a_img%s' % (entry)])



if __name__ == '__main__':
    main()
    from soma.qt_gui.qt_backend import Qt
    Qt.qApp.exec_()
