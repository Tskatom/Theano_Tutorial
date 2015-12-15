import numpy as np
import time
import sys
import subprocess
import os
import random

from elman import model


def main():
    s = {
        "fold":3,
        "lr": 0.06,
        "verbose": 1,
        "decay": False,
        "win": 7,
        "bs": 9, # number of backprop through time steps
        "nhidden": 100,
        "seed": 345,
        "emb_dimension":100,
        'nepochs': 50
    }

    folder = os.path.basename(__file__).split('.')[0]
    if not os.path.exists(folder):
        os.mkdir(folder)

    