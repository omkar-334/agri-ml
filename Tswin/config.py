# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = [""]

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 32
# _C.DATA.DATA_PATH = ""
# _C.DATA.DATASET = "ImageD2"
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = "bicubic"
# _C.DATA.ZIP_MODE = False
# _C.DATA.CACHE_MODE = "part"
# _C.DATA.PIN_MEMORY = True
# _C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = "cswin_boat"
_C.MODEL.NAME = "CSWin_BOAT_64_24322_small_224"
# _C.MODEL.RESUME = ""
_C.MODEL.DROP_RATE = 0.0
_C.MODEL.DROP_PATH_RATE = 0.1
_C.MODEL.LABEL_SMOOTHING = 0.1

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.0
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False  # absolute position embedding
_C.MODEL.SWIN.RPE = True  # reletive position embedding
_C.MODEL.SWIN.PATCH_NORM = True

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 35
_C.TRAIN.WARMUP_EPOCHS = 5
_C.TRAIN.WEIGHT_DECAY = 0.005
_C.TRAIN.BASE_LR = 2e-4
_C.TRAIN.WARMUP_LR = 2e-6
_C.TRAIN.MIN_LR = 2e-4
_C.TRAIN.CLIP_GRAD = 5.0
# _C.TRAIN.AUTO_RESUME = True
_C.TRAIN.ACCUMULATION_STEPS = 0
# _C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = "cosine"
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = "adamw"
_C.TRAIN.OPTIMIZER.EPS = 1e-8  # optimizer epsilon
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9  # SGD Momentum


# Pretraining
_C.TRAIN.USE_DRLOC = False
_C.TRAIN.LAMBDA_DRLOC = 0.20
_C.TRAIN.SAMPLE_SIZE = 64
_C.TRAIN.USE_NORMAL = False
_C.TRAIN.DRLOC_MODE = "l1"
_C.TRAIN.USE_ABS = False
_C.TRAIN.SSL_WARMUP_EPOCHS = 20
_C.TRAIN.USE_MULTISCALE = False

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
_C.AUG.COLOR_JITTER = 0.4
_C.AUG.AUTO_AUGMENT = "rand-m9-mstd0.5-inc1"  # "v0" or "original"
_C.AUG.REPROB = 0.25
_C.AUG.REMODE = "pixel"
_C.AUG.RECOUNT = 1
_C.AUG.MIXUP = 0.8
_C.AUG.CUTMIX = 1.0
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
# _C.AUG.MIXUP_MODE = "batch"

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.CROP = True  # Whether to use center crop when testing

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.AMP_OPT_LEVEL = ""
_C.OUTPUT = "output"
_C.TAG = "default"
_C.SAVE_FREQ = 5
_C.PRINT_FREQ = 10
_C.SEED = 0
_C.EVAL_MODE = False
_C.THROUGHPUT_MODE = False
_C.LOCAL_RANK = 0


def get_config():
    """Return the default config as a frozen CfgNode object."""
    return _C.clone()
