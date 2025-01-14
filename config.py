# --------------------------------------------------------
# Configurations for domain adaptation
# Copyright (c) 2019 valeo.ai
#
# Written by Tsung-Yu Lin
# Adapted from https://github.com/valeoai/ADVENT
# --------------------------------------------------------

import os.path as osp
from yacs.config import CfgNode as CN

_C = CN()

# COMMON CONFIGS
# source domain
_C.SOURCE = 'Zong'
# target domain
_C.TARGET = 'Chang_Gung'
# Number of workers for dataloading
_C.NUM_WORKERS = 4
# Directories
_C.DATASET = CN()
_C.DATASET.IMGSOURCE = r""
_C.DATASET.MASKSOURCE = r""
_C.DATASET.IMGTARGET = r""
_C.DATASET.IMGTEST = r""
_C.DATASET.MASKTEST = r""
# Number of object classes
_C.NUM_CLASSES = 2
# Exp dirs
_C.EXP_NAME = ''
_C.EXP_ROOT = 'experiments'
_C.EXP_ROOT_SNAPSHOT = osp.join(_C.EXP_ROOT, 'snapshots')
_C.EXP_ROOT_LOGS = osp.join(_C.EXP_ROOT, 'logs')
# CUDA
_C.GPU_ID = 0
# TRAIN CONFIGS
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE_SOURCE = 1
_C.TRAIN.BATCH_SIZE_TARGET = 1
_C.TRAIN.IGNORE_LABEL = 255
_C.TRAIN.INPUT_SIZE_SOURCE = 257
_C.TRAIN.INPUT_SIZE_TARGET = 257
# Segmentation network params
_C.TRAIN.MODEL = 'DeepLabv2'
_C.TRAIN.MULTI_LEVEL = True
_C.TRAIN.RESTORE_FROM = ''
_C.TRAIN.LEARNING_RATE = 2.5e-4
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 0.0005
_C.TRAIN.POWER = 0.9
_C.TRAIN.LAMBDA_SEG_MAIN = 1.0
_C.TRAIN.LAMBDA_SEG_AUX = 0.1  # weight of conv4 prediction. Used in multi-level setting.
# Domain adaptation
_C.TRAIN.DA_METHOD = 'AdvEnt'
# Adversarial training params
_C.TRAIN.LEARNING_RATE_D = 1e-4
_C.TRAIN.LAMBDA_ADV_MAIN = 0.001
_C.TRAIN.LAMBDA_ADV_AUX = 0.0002
# MinEnt params
_C.TRAIN.LAMBDA_ENT_MAIN = 0.001
_C.TRAIN.LAMBDA_ENT_AUX = 0.0002
# Other params
_C.TRAIN.MAX_ITERS = 250000
_C.TRAIN.EARLY_STOP = 120000
_C.TRAIN.SAVE_PRED_EVERY = 5
_C.TRAIN.SNAPSHOT_DIR = ""
_C.TRAIN.RANDOM_SEED = 1234
_C.TRAIN.TENSORBOARD_LOGDIR = ''
_C.TRAIN.TENSORBOARD_VIZRATE = 100
# TEST CONFIGS
_C.TEST = CN()
_C.TEST.MODE = 'best'  # {'single', 'best'}
# model
_C.TEST.MODEL = ('DeepLabv2',)
_C.TEST.MODEL_WEIGHT = (1.0,)
_C.TEST.MULTI_LEVEL = (True,)
_C.TEST.RESTORE_FROM = ['',]
_C.TEST.SNAPSHOT_DIR = ['',]  # used in 'best' mode
_C.TEST.SNAPSHOT_STEP = 1  # used in 'best' mode
_C.TEST.SNAPSHOT_MAXITER = 1000  # used in 'best' mode
# Test sets
_C.TEST.BATCH_SIZE_TARGET = 1
_C.TEST.INPUT_SIZE_TARGET = 257
_C.TEST.OUTPUT_SIZE_TARGET = 257
_C.TEST.WAIT_MODEL = True

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`

