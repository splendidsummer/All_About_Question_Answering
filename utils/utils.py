import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, json, os, random
from mpl_toolkits.axes_grid1 import ImageGrid
import shutil, wandb, torch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

