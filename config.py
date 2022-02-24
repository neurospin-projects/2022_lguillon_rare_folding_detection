class Config:

    def __init__(self):
        self.model = 'vae'
        self.batch_size = 64
        self.nb_epoch = 300 #300
        self.n = 75
        self.kl = 8
        self.weight = 2
        self.lr = 1e-4
        self.in_shape = (1, 40, 40, 56) # input size with padding
        self.min_size = 300
        self.save_dir = f"/neurospin/dico/lguillon/miccai_22/analyses_gridsearch_clf/n_{self.n}_kl_{self.kl}/"
        self.data_dir = '/neurospin/dico/data/deep_folding/current/crops/SC/' \
                         'mask/sulcus_based/2mm/'
        self.subject_dir = "/neurospin/dico/lguillon/miccai_22/data/train_list.csv"
        self.benchmark_list = "/neurospin/dico/lguillon/miccai_22/data/benchmark_list.csv"
        self.benchmark_dir_1 = '/neurospin/dico/data/deep_folding/current/crops/PRECENTRAL/' \
                         'mask/sulcus_based/2mm/'
        self.one_handed_dir = '/neurospin/dico/data/deep_folding/current/crops/SC/' \
                         'mask/sulcus_based/2mm/one_handed_dataset/'
        self.benchmark_dir_2 = '/neurospin/dico/data/deep_folding/current/crops/POSTCENTRAL/' \
                         'mask/sulcus_based/2mm/'
