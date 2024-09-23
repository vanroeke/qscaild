import os
import sys
import os.path
import shutil
import subprocess
import glob
import pprint
import datetime
import logging
import calculator
import mlip2vasp
import time
import numpy as np

#Call with ./gather_runs.py MLIP_train_set direc1 .. direcn
MLIP_train_set = sys.argv[1]
for direc in sys.argv[2:]:
    for i in [ d for d in os.listdir(direc) if "config" in d ]:
        calculator.add_to_train(os.path.join(direc,i),MLIP_train_set)
