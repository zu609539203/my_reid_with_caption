class Config(object):
    def __init__(self):

        # Learning Rates
        self.lr_backbone = 1e-5#视觉骨干网络的学习率很小
        self.lr = 1e-4#transformer的学习率较大

        # Epochs
        self.epochs = 30
        self.lr_drop = 20
        self.start_epoch = 0
        self.weight_decay = 1e-4

        # Backbone
        self.backbone = 'resnet101'
        self.position_embedding = 'sine'
        self.dilation = True
        
        # Basic
        self.device = 'cuda:0'
        self.seed = 42
        self.batch_size = 3 # 16
        self.num_workers = 0
        # self.checkpoint_preTrain = 'C:/Users/bing/.cache/torch/h ub/checkpoints/weight493084032.pth'
        self.checkpoint_preTrain = ''
        self.checkpoint_finetune = './checkpoints/CUHK_finetune.pth'
        # self.checkpoint_finetune = ''
        self.checkpoint_onlyCHUK = './checkpoints/CUHK_only.pth'
        # self.checkpoint_onlyCHUK = ''
        self.clip_max_norm = 0.1

        # Transformer
        self.hidden_dim = 256
        self.pad_token_id = 0
        self.max_position_embeddings = 128
        self.layer_norm_eps = 1e-12
        self.dropout = 0.1
        self.vocab_size = 30522

        self.enc_layers = 6
        self.dec_layers = 6
        self.dim_feedforward = 2048
        self.nheads = 8
        self.pre_norm = True

        # Dataset
        self.dir = '../coco'
        self.CUHKdata = '../CUHK-PEDES/imgs'
        self.CUHKano = '../CUHK-PEDES'
        self.limit = -1