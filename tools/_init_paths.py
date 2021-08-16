"""Add {PROJECT ROOT}/datasets. to PYTHONPATH

Usage:
import this module before import any modules under lib/
e.g.
    import _init_paths
    from datasets.TripletDataLoader import TripletDataLoader
"""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.abspath(osp.dirname(osp.dirname(__file__)))

# Add lib to PYTHONPATH
lib_dir = osp.join(this_dir, 'lib')
add_path(this_dir)
add_path(lib_dir)
