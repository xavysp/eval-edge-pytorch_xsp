import os
import numpy as np
from itertools import product
from argparse import ArgumentParser
from os.path import join

from nms_process import nms_process
from impl.edges_eval_dir import edges_eval_dir


def eval_one_epoch(args):

    print(args.root)
    result_dir = join(args.root, "png")  # forward result directory, if it is on mat "mat"
    nms_dir = join(args.root, "nms")  # forward result directory
    edge_size = 1.1 # default UDED = 1.1, default = 1.01
    datasets = {
        "BSDS": "GT/BSDS",
        "BIPED": "GT/BIPED",
        "NYUD": "GT/NYUD",
        "UDED": "GT/UDED",
        "TEEDedges": "GT/TEEDedges"
    }

    gt_dir = datasets[args.dataset]

    key = args.key  # x = scipy.io.loadmat(filename)[key]
    file_format = args.file_format  # ".mat", ".npy" or ".png"

    thrs = 99 if args.full else 9
    print("result dir: ", result_dir)
    nms_process(result_dir, nms_dir, key, file_format,edge_size)

    edges_eval_dir(nms_dir, gt_dir, thrs=thrs, thin=1, max_dist=0.0075,
               # numerical values depend on computational capacity
               workers=4,           # per-threshold
               workers_img=8)       # per-image

if __name__ == '__main__':
    parser = ArgumentParser("edge eval")
    parser.add_argument("root", type=str, default="examples/hed_result", help="results directory")
    parser.add_argument("--key", type=str, default="groundTruth", help="key")
    parser.add_argument("--file_format", type=str, default=".mat", help=".mat or .npy")
    parser.add_argument("--workers", type=int, default="-1", help="number workers, -1 for all workers")
    parser.add_argument("--dataset")
    parser.add_argument("-f","--full",action="store_true")
    args = parser.parse_args()

    eval_one_epoch(args)
