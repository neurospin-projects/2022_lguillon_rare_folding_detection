class Config:

    def __init__(self):
        self.batch_size = 4
        self.nb_epoch = 2 #300
        self.n = 4
        self.lr = 2e-4
        self.in_shape = (1, 40, 40, 56) # input size with padding
        self.min_size = 300
        self.save_dir = "/neurospin/dico/lguillon/miccai_22/"
        self.data_dir = '/neurospin/dico/data/deep_folding/current/crops/SC/' \
                         'mask/sulcus_based/2mm/'
        self.subject_dir = "/neurospin/dico/data/deep_folding/current/HCP_half_1bis.csv"
