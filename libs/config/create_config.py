## Original project
# Project: squeezeDetOnKeras
# Filename: visualization
# Author: Christopher Ehmann
# Date: 12.12.17
# Organisation: searchInk
# Email: christopher@searchink.com

## Edited project
# Author: Jordy A. Faria de Araújo
# Date: 25/07/2018
# Email: jordyfaria0@gmail.com
# Github: AjJordy


""" Model configuration for COCO 2017 dataset """
import numpy as np
from easydict import EasyDict as edict
import json
import argparse


def squeezeDet_config(name):
    """Specify the parameters to tune below."""
    cfg = edict()

    cfg.init_file = 'none'
    # cfg.init_file = 'log\\imagenet.h5' 

    if cfg.init_file != 'none':       
        """ Dataset imagetagger """
        # cfg.CLASS_BITBOT =  {'goalpost':0,'ball':1,'goal':2,'robot':3,'obstacle':4,'line':5}
        # cfg.CLASS_NAMES = ['goalpost','ball','goal','robot','obstacle','line']
        # cfg.IDS = [0,1,2,3,4,5]
        # cfg.CLASSES = 6 
        # print("init_file")

        """ Dataset imagetagger with only ball"""
        cfg.CLASS_BITBOT =  {'ball':0}
        cfg.CLASS_NAMES = ['ball']
        cfg.IDS = [0]
        cfg.CLASSES = 1
        print("init_file only ball")       

    else:
        """ Dataset COCO full """ 
        # cfg.CLASS_NAMES =  ['person', 'bicycle', 'car', 'motorcycle', 'airplane','bus', 'train', 'truck', 'boat', 'traffic_light','fire_hydrant', 'stop_sign', 'parking_meter', 
        #                     'bench', 'bird','cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear','zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
        #                     'frisbee', 'skis', 'snowboard', 'sports_ball','kite', 'baseball_bat', 'baseball_glove', 'skateboard','surfboard', 'tennis_racket', 'bottle', 
        #                     'wine_glass', 'cup','fork', 'knife', 'spoon', 'bowl', 'banana', 'apple','sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza','donut',
        #                     'cake', 'chair', 'couch', 'potted_plant', 'bed','dining_table', 'toilet', 'tv', 'laptop', 'mouse', 'remote','keyboard', 'cell_phone', 'microwave', 
        #                     'oven', 'toaster','sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors','teddy_bear', 'hair_drier', 'toothbrush']
        # cfg.IDS = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,
        #            58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]
        # cfg.CLASS_ID = {'1':'person','2':'bicycle','3': 'car','4':'motorcycle','5': 'airplane','6':'bus','7':'train','8':'truck','9':'boat','10':'traffic_light',
        #                 '11':'fire_hydrant','13':'stop_sign','14':'parking_meter','15':'bench','16':'bird','17':'cat','18':'dog','19':'horse','20':'sheep',
        #                 '21':'cow','22':'elephant','23':'bear','24':'zebra','25':'giraffe','27':'backpack','28':'umbrella','31':'handbag','32':'tie','33':'suitcase',
        #                 '34':'frisbee','35':'skis','36':'snowboard','37':'sports_ball','38':'kite','39':'baseball_bat','40':'baseball_glove','41':'skateboard',
        #                 '42':'surfboard','43':'tennis_racket','44':'bottle','46':'wine_glass','47':'cup','48':'fork','49':'knife','50':'spoon','51':'bowl',
        #                 '52':'banana','53':'apple','54':'sandwich','55':'orange','56':'broccoli','57':'carrot','58':'hot_dog','59':'pizza','60':'donut','61':'cake',
        #                 '62':'chair','63':'couch','64':'potted_plant','65':'bed','67':'dining_table','70':'toilet','72':'tv','73':'laptop','74':'mouse','75':'remote',
        #                 '76':'keyboard','77':'cell_phone','78':'microwave','79':'oven','80':'toaster','81':'sink','82':'refrigerator','84':'book','85':'clock',
        #                 '86':'vase','87':'scissors','88':'teddy_bear','89':'hair_drier','90':'toothbrush'}
        # # number of categories to classify
        # # it's 91 because the max value of the IDS is 90, but we have only 80 classes
        # cfg.CLASSES = 91 # len(cfg.CLASS_NAMES) == 80
        # print("COCO")
              

        """ Dataset COCO smaller """
        # cfg.CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle','sports_ball','laptop','mouse','remote','cell_phone','microwave','toaster','refrigerator','hair_drier']
        # # cfg.IDS = [1,2,3,4,37,73,74,75,77,78,80,82,89]
        # cfg.IDS = [0,1,2,3,4,5,6,7,8,9,10,11,12]
        # cfg.CLASS_ID = {'0':'person','1':'bicycle','2': 'car','3':'motorcycle','4':'sports_ball','5':'laptop','6':'mouse','7':'remote','8':'cell_phone','9':'microwave',
        #                 '10':'toaster','11':'refrigerator','12':'hair_drier'}
        # cfg.CLASSES = 13
        # print("Small COCO")
        # cfg.CLASS_ID = {'1':'person','2':'bicycle','3': 'car','4':'motorcycle','37':'sports_ball','73':'laptop','74':'mouse','75':'remote','77':'cell_phone',
        #                 '78':'microwave','80':'toaster','82':'refrigerator','89':'hair_drier'} 


        """ Dataset ImageTagger with only ball """
        cfg.CLASS_BITBOT =  {'ball':0}
        cfg.CLASS_NAMES = ['ball']
        cfg.IDS = [0]
        cfg.CLASSES = 1
        print("Only ball")
    
    # classes to class index dict
    cfg.CLASS_TO_IDX = dict(zip(cfg.CLASS_NAMES, cfg.IDS))

    # Probability to keep a node in dropout
    cfg.KEEP_PROB = 0.5

    # a small value used to prevent numerical instability
    cfg.EPSILON = 1e-16

    # threshold for safe exponential operation
    cfg.EXP_THRESH = 1.0

    #image properties
    # (768 * 624) / (78 * 24) = 256
    # (256 * 256) / (16 * 16) = 256
    cfg.IMAGE_WIDTH = 256 # 768 # 1248 
    cfg.IMAGE_HEIGHT = 256 # 624 # 384  
    cfg.N_CHANNELS = 3

    #batch sizes
    cfg.BATCH_SIZE = 64 # 8 
    cfg.VISUALIZATION_BATCH_SIZE = 64 # 8

    #SGD + Momentum parameters
    cfg.WEIGHT_DECAY = 0.001
    cfg.LEARNING_RATE = 0.01
    cfg.MAX_GRAD_NORM = 1.0
    cfg.MOMENTUM = 0.9

    #coefficients of loss function
    cfg.LOSS_COEF_BBOX = 20.0
    cfg.LOSS_COEF_CONF_POS = 75.0
    cfg.LOSS_COEF_CONF_NEG = 100.0
    cfg.LOSS_COEF_CLASS = 1.0

    #thesholds for evaluation
    cfg.NMS_THRESH = 0.4
    cfg.PROB_THRESH = 0.005
    cfg.TOP_N_DETECTION = 1 # 64
    cfg.IOU_THRESHOLD = 0.5
    cfg.FINAL_THRESHOLD = 0.0

    cfg.ANCHOR_SEED = np.array([[  36.,  37.], [ 366., 174.], [ 115.,  59.],
                                [ 162.,  87.], [  38.,  90.], [ 258., 173.],
                                [ 224., 108.], [  78., 170.], [  72.,  43.]])

    cfg.ANCHOR_PER_GRID = len(cfg.ANCHOR_SEED)
    cfg.ANCHORS_WIDTH = 16 # 78   
    cfg.ANCHORS_HEIGHT = 16 # 24        

    return cfg


def create_config_from_dict(dictionary = {}, name="squeeze.config"):
    """Creates a config and saves it

    Keyword Arguments:
        dictionary {dict} -- [description] (default: {{}})
        name {str} -- [description] (default: {"squeeze.config"})
    """

    cfg = squeezeDet_config(name)
    for key, value in dictionary.items():
        cfg[key] = value

    save_dict(cfg, name)

#save a config files to json
def save_dict(dict, name="squeeze.config"):

    #change np arrays to lists for storing
    for key, val, in dict.items():
        if type(val) is np.ndarray:
            dict[key] = val.tolist()

    with open(name, "w") as f:
        json.dump(dict, f, sort_keys=True, indent=0 )  ### this saves the array in .json format


def load_dict(path):
    """Loads a dictionary from a given path name

    Arguments:
        path {[type]} -- string of path

    Returns:
        [type] -- [description]
    """

    with open(path, "r") as f:
        cfg = json.load(f)  ### this loads the array from .json format

    #changes lists back
    for key, val, in cfg.items():
        if type(val) is list:
            cfg[key] = np.array(val)

    #cast do easydict
    cfg = edict(cfg)

    #create full anchors from seed
    cfg.ANCHOR_BOX, cfg.N_ANCHORS_HEIGHT, cfg.N_ANCHORS_WIDTH = set_anchors(cfg)
    cfg.ANCHORS = len(cfg.ANCHOR_BOX)

    #if you added a class in the config manually, but were to lazy to update
    # cfg.CLASSES = len(cfg.CLASS_NAMES)
    # cfg.CLASS_TO_IDX = dict(zip(cfg.CLASS_NAMES, cfg.IDS))

    return cfg


#compute the anchors for the grid from the seed
def set_anchors(cfg):
  H, W, B = cfg.ANCHORS_HEIGHT, cfg.ANCHORS_WIDTH, cfg.ANCHOR_PER_GRID

  anchor_shapes = np.reshape([cfg.ANCHOR_SEED] * H * W,(H, W, B, 2))
  center_x = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, W+1)*float(cfg.IMAGE_WIDTH)/(W+1)]*H*B),
              (B, H, W)
          ),
          (1, 2, 0)
      ),
      (H, W, B, 1)
  )
  center_y = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, H+1)*float(cfg.IMAGE_HEIGHT)/(H+1)]*W*B),
              (B, W, H)
          ),
          (2, 1, 0)
      ),
      (H, W, B, 1)
  )
  anchors = np.reshape(np.concatenate((center_x, center_y, anchor_shapes),axis=3),(-1, 4))

  return anchors, H, W




if __name__ == "__main__":

  # parse arguments
  parser = argparse.ArgumentParser(description='Creates config file for squeezeDet training')
  parser.add_argument("--name", help="Name of the config file. DEFAULT: squeeze.config")
  args = parser.parse_args()

  name = "squeeze.config"

  if args.name:
      name = args.name

  cfg = squeezeDet_config(name=name)


  save_dict(cfg, name)
