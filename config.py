class Config:

    def __init__(self):
        self.model = 'vae'
        self.batch_size = 8
        self.nb_epoch = 150 #300
        self.n = 75
        self.kl = 2
        self.weight = 2
        self.lr = 5e-4
        #self.in_shape = (1, 80, 72, 96) # input size with padding
        #### FOR SC:
        #self.in_shape = (1, 80, 80, 96) # input size with padding
        #### FOR CINGULATE RIGHT:
        #self.in_shape = (1, 40, 128, 128) # input size with padding
        #### FOR CINGULATE LEFT:
        #self.in_shape = (1, 32, 136, 112) # input size with padding
        #self.min_size = 300
        #### FOR SC + precentral RIGHT:
        self.in_shape = (1, 80, 88, 104) # input size with padding

        #### FOR SC
        # self.save_dir = f"/neurospin/dico/lguillon/distmap/rotation_-3_3/"
        # self.data_dir = '/neurospin/dico/data/deep_folding/current/datasets/' \
        #                 'hcp/crops/1mm/SC/no_mask/Rdistmaps'
        # self.aug_dir = '/neurospin/dico/data/deep_folding/current/datasets/hcp/crops/1mm/SC/no_mask'

        #### FOR CCD
        #self.data_dir = '/neurospin/dico/lguillon/distmap/data/'
        #self.data_dir = '/neurospin/dico/lguillon/distmap/CCD/data/'
        #self.save_dir = f"/neurospin/dico/lguillon/distmap/CCD/runs/left/seed_10/"
        #self.aug_dir = '/neurospin/dico/data/deep_folding/current/datasets/hcp/crops/1mm/CCD/no_mask'

        #### FOR PBS
        self.data_dir = '/neurospin/dico/lguillon/distmap/PBS/data/'
        self.aug_dir = '/neurospin/dico/data/deep_folding/current/datasets/'\
                        'hcp/crops/1mm/SC_precentral/no_mask'
        self.save_dir = f"/neurospin/dico/lguillon/distmap/PBS/runs/right/"

        self.subject_dir = "/neurospin/dico/lguillon/distmap/data/train_list.csv"
        #self.subject_dir = "/neurospin/dico/lguillon/distmap/CCD/data/train_list.csv"

        # self.side = 'R'
        # self.sulci_list = ['S.C.']
        # self.mask_dir = '/neurospin/dico/data/deep_folding/current/datasets/' \
        #                 'hcp/mask'

        self.benchmark_list = "/neurospin/dico/lguillon/miccai_22/data/benchmark_list.csv"

        self.benchmark_dir_1 = '/neurospin/dico/data/deep_folding/current/' \
                        'datasets/hcp/crops/1mm/precentral/no_mask/benchmark/'

        # self.one_handed_dir = '/neurospin/dico/data/deep_folding/current/crops/SC/' \
        #                  'mask/sulcus_based/2mm/one_handed_dataset/'

        self.benchmark_dir_2 = '/neurospin/dico/data/deep_folding/current/' \
                        'datasets/hcp/crops/1mm/postcentral/no_mask/benchmark'
