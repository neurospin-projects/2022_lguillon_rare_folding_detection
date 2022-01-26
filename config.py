class Config:

    def __init__(self):
        self.batch_size = 64
        self.nb_epoch = 2 #300
        self.n = 4
        self.lr = 2e-4
        self.in_shape = (1, 20, 40, 40) # input size with padding
        self.save_dir = "/neurospin/dico/lguillon/micai_22/"
        self.data_dir = '/neurospin/dico/data/deep_folding/current/crops/SC/' \
                         'mask/sulcus_based/2mm/'
        self.subject_dir = "/neurospin/dico/data/deep_folding/current/HCP_half_1bis.csv"
