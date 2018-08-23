import subprocess
import os
import numpy as np
import sys

sys.setrecursionlimit(5000)
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

env = os.environ
if subprocess.call(['make', '-C', BASE_DIR], shell=True, env={'PATH': os.getenv('PATH')}) != 0:  # return value
    raise RuntimeError('Cannot compile lanms: {}'.format(BASE_DIR))


def merge_quadrangle_n9(polys, thres=0.3, precision=10000):
    from east_detection.lanms import merge_quadrangle_n9 as nms_impl

    if len(polys) == 0:
        return np.array([], dtype='float64')

    p = polys.copy()
    p[:, :8] *= precision
    ret = np.array(nms_impl(p, thres), dtype='float64')
    ret[:, :8] /= precision
    return ret

