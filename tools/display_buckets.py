""" The aim of this script is to display saved model outputs thanks to Anatomist.
Model outputs are stored as numpy arrays.
"""

import anatomist.api as anatomist
#!/usr/bin/env python

import anatomist.api as ana
from soma.qt_gui.qt_backend import Qt

#print(Qt.QApplication.instance())

run_gui = Qt.QApplication.instance() is None

a = ana.Anatomist()

# General paths
path_hcp = '/neurospin/dico/data/deep_folding/current/crops/SC/mask/sulcus_based/2mm/Rbuckets/'

path_oh = '/neurospin/dico/data/deep_folding/current/crops/SC/mask/sulcus_based/2mm/one_handed_dataset/Rbuckets/'

#vblock = a.createWindowsBlock(nbRows=1, nbCols=6)
# Load objects
bucket1 = a.loadObject(path_hcp + f"100206_cropped_skeleton.bck")
bucket2 = a.loadObject(path_hcp + f"111009_cropped_skeleton.bck")
bucket3 = a.loadObject(path_hcp + f"751550_cropped_skeleton.bck")
bucket4 = a.loadObject(path_hcp + f"843151_cropped_skeleton.bck")
bucket5 = a.loadObject(path_hcp + f"210011_cropped_skeleton.bck")
bucket6 = a.loadObject(path_oh + f"MA03_struct_nf_cropped_skeleton.bck")
bucket7 = a.loadObject(path_oh + f"MA12_struct_nf_cropped_skeleton.bck")
bucket8 = a.loadObject(path_oh + f"PA02_struct_cropped_skeleton.bck")
bucket9 = a.loadObject(path_oh + f"PA17_struct_cropped_skeleton.bck")
bucket10 = a.loadObject(path_oh + f"PA30_struct_cropped_skeleton.bck")
#for k in range(6):
#    bucket1 = a.loadObject(path + f"interpolation_bucket_step_{k}.bck")

#list_tex = [constel_tex1, constel_tex2, constel_tex3, constel_tex4, constel_tex5, constel_tex6, constel_tex7, constel_tex8, constel_tex9, constel_tex10]

# Specify palette and interpolation
#a.execute('TexturingParams', objects=list_tex, interpolation='rgb')
#a.setObjectPalette(list_tex, palette='parcellation720', minVal=0., maxVal=1.)

# Init windows
vblock = a.createWindowsBlock(nbRows=1, nbCols=6)
w1 = a.createWindow('3D', block=vblock)
w2 = a.createWindow('3D', block=vblock)
w3 = a.createWindow('3D', block=vblock)
w4 = a.createWindow('3D', block=vblock)
w5 = a.createWindow('3D', block=vblock)
w6 = a.createWindow('3D', block=vblock)
w7 = a.createWindow('3D', block=vblock)
w8 = a.createWindow('3D', block=vblock)
w9 = a.createWindow('3D', block=vblock)
w10 = a.createWindow('3D', block=vblock)

# Add fusion objects to windows
w1.addObjects(bucket1)
w2.addObjects(bucket2)
w3.addObjects(bucket3)
w4.addObjects(bucket4)
w5.addObjects(bucket5)
w6.addObjects(bucket6)
w7.addObjects(bucket7)
w8.addObjects(bucket8)
w9.addObjects(bucket9)
w10.addObjects(bucket10)

w1.windowConfig(cursor_visibility=0)
w2.windowConfig(cursor_visibility=0)
w3.windowConfig(cursor_visibility=0)
w4.windowConfig(cursor_visibility=0)
w5.windowConfig(cursor_visibility=0)
w6.windowConfig(cursor_visibility=0)
w7.windowConfig(cursor_visibility=0)
w8.windowConfig(cursor_visibility=0)
w9.windowConfig(cursor_visibility=0)
w10.windowConfig(cursor_visibility=0)


# Set camera
w1.camera(view_quaternion=[0.361454546451569, -0.398135215044022, -0.458537578582764, 0.707518458366394], zoom=1.)
w2.camera(view_quaternion=[0.361454546451569, -0.398135215044022, -0.458537578582764, 0.707518458366394], zoom=1.)
w3.camera(view_quaternion=[0.361454546451569, -0.398135215044022, -0.458537578582764, 0.707518458366394], zoom=1.)
w4.camera(view_quaternion=[0.361454546451569, -0.398135215044022, -0.458537578582764, 0.707518458366394], zoom=1.)
w5.camera(view_quaternion=[0.361454546451569, -0.398135215044022, -0.458537578582764, 0.707518458366394], zoom=1.)
w6.camera(view_quaternion=[0.361454546451569, -0.398135215044022, -0.458537578582764, 0.707518458366394], zoom=1.)
w7.camera(view_quaternion=[0.361454546451569, -0.398135215044022, -0.458537578582764, 0.707518458366394], zoom=1.)
w8.camera(view_quaternion=[0.361454546451569, -0.398135215044022, -0.458537578582764, 0.707518458366394], zoom=1.)
w9.camera(view_quaternion=[0.361454546451569, -0.398135215044022, -0.458537578582764, 0.707518458366394], zoom=1.)
w10.camera(view_quaternion=[0.361454546451569, -0.398135215044022, -0.458537578582764, 0.707518458366394], zoom=1.)


a.execute('LinkedCursor', window=w1, position=[83.0825, 23.6431, 124.721])
# a.execute('LinkWindows', windows=[w1])
# group = a.execute('LinkWindows', windows=[w2])

if run_gui:
    Qt.qApp.exec_()
