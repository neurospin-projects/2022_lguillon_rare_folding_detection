class Config:

    def __init__(self):
        self.model = 'vae'
        self.batch_size = 8
        self.nb_epoch = 150 #300
        self.n = 75
        self.kl = 2
        self.weight = 2
        self.lr = 1e-4
        #self.in_shape = (1, 80, 72, 96) # input size with padding
        self.in_shape = (1, 80, 80, 96) # input size with padding
        self.min_size = 300

        self.save_dir = f"/neurospin/dico/lguillon/distmap/skel_comparison/w5_1e-4_rot_bs_32_23/"
        #self.data_dir = '/neurospin/dico/data/deep_folding/current/datasets/' \
        #                'hcp/crops/1mm/SC/mask/'
        self.data_dir = '/neurospin/dico/lguillon/distmap/data/'
        self.subject_dir = "/neurospin/dico/lguillon/distmap/data/train_list.csv"

        self.src_dir = '/neurospin/dico/data/deep_folding/current/datasets/' \
                        'hcp/crops/1mm/SC/no_mask'
        self.aug_dir = '/neurospin/dico/data/deep_folding/current/datasets/hcp/crops/1mm/SC/no_mask'

        self.side = 'R'
        self.sulci_list = ['S.C.']
        self.mask_dir = '/neurospin/dico/data/deep_folding/current/datasets/' \
                        'hcp/mask'

        self.benchmark_list = "/neurospin/dico/lguillon/miccai_22/data/benchmark_list.csv"

        self.benchmark_dir_1 = '/neurospin/dico/data/deep_folding/current/' \
                        'datasets/hcp/crops/1mm/precentral/no_mask/benchmark/'

        self.one_handed_dir = '/neurospin/dico/data/deep_folding/current/crops/SC/' \
                         'mask/sulcus_based/2mm/one_handed_dataset/'

        self.benchmark_dir_2 = '/neurospin/dico/data/deep_folding/current/' \
                        'datasets/hcp/crops/1mm/postcentral/no_mask/benchmark'