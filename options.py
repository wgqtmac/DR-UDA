import argparse
import os
from util import util
import torch

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParse(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', required=True, \
            help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--base_lr', type=float, default=0.01, \
            help='base learning rate')
        self.paser.add_argument('--momentum', type=float, default=0.9, \
            help='momentum')
