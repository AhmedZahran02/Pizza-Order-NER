
import pip
import os
def import_or_install(package, download):
    try:
        __import__(package)
        print("Package (" + package + ") found importing ..." )
    except ImportError:
        print("Package (" + package + ") not found do you want to download it ? (y/n)")
        answer = input()

        if answer[0] == "y":
            print("installing .... (if failed, please try to download it using powershell or raising administration)")
            pip.main(['install', download])

            import_or_install(package, download)
        else:
            print("Exiting ...")
            quit(400)

required_packages = [
    ("json", "json"),
    ("re", "regex"),
    ("random", "random"),
    ("pandas", "pandas"),
    ("nltk", "nltk"),
    ("torch", "torch"),
    ("pickle", "pickle"),
    ("tqdm", "tqdm"),
    ("rich", "rich"),
]

for package, download in required_packages:
    import_or_install(package, download)
os.system("cls")

import json
import re
import random
import pandas as pd

from nltk.stem import PorterStemmer

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pickle import load


from tqdm import tqdm
from rich.console import Console
from rich.columns import Columns
from rich.panel import Panel

from var import *
from classes import RNNSequenceLabeling