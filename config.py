class Config:

    def __init__(self):
        self.model = 'vae'
        self.batch_size = 8
        self.nb_epoch = 2 #300
        self.n = 40
        self.kl = 2
        self.weight = 2
        self.lr = 1e-4
        self.in_shape = (1, 80, 56, 80) # input size with padding
        self.min_size = 300

        self.save_dir = f"/neurospin/dico/lguillon/distmap/"
        self.data_dir = '/neurospin/dico/data/deep_folding/current/datasets/' \
                        'hcp/crops/1mm/SC/mask/'
        self.subject_dir = "/neurospin/dico/lguillon/miccai_22/data/train_list.csv"

        self.aug_dir = '/neurospin/dico/data/deep_folding/current/datasets/' \
                       'hcp/intermediate_crops_aug/'
        self.side = 'R'
        self.sulci_list = ['S.C.']
        self.mask_dir = '/neurospin/dico/data/deep_folding/current/datasets/' \
                        'hcp/mask'

        self.benchmark_list = "/neurospin/dico/lguillon/miccai_22/data/benchmark_list.csv"

        self.benchmark_dir_1 = '/neurospin/dico/data/deep_folding/current/crops/PRECENTRAL/' \
                         'mask/sulcus_based/2mm/'
        self.one_handed_dir = '/neurospin/dico/data/deep_folding/current/crops/SC/' \
                         'mask/sulcus_based/2mm/one_handed_dataset/'
        self.benchmark_dir_2 = '/neurospin/dico/data/deep_folding/current/crops/POSTCENTRAL/' \
                         'mask/sulcus_based/2mm/'
