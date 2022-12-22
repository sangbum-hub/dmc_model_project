import os 
import torch 
import numpy as np 
import random 
import glob 

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
FILE_PATH = glob.glob(f'{DATA_DIR}/*.csv')
PARAM_PATH = os.path.join(BASE_DIR, 'model_parameter')

def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
    