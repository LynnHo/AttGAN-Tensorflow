from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys


def add_path(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)


def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)


if __name__ == '__main__':
    pass
