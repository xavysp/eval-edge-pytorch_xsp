import os, sys
import numpy as np
from itertools import product
from argparse import ArgumentParser
from os.path import join

from nms_process import nms_process
from impl.edges_eval_dir import edges_eval_dir


def eval_one_epoch(args):
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
    if not os.path.exists(gt_dir):
        gt_dir = join(gt_dir , "gt")
        print(f"** There is not {gt_dir} folder, we've created for you, restart!**")
        os.makedirs(gt_dir)
        sys.exit()
    if len(next(os.walk(gt_dir))[1])>0:
        tmp_dirs = os.listdir(gt_dir)
        gt_dir = os.path.join(gt_dir, args.gt_dir) # only choose thye just gt in the list
        print(f"Dataset starting to evaluating on {gt_dir}!")
    else:
        gt_dir = os.path.join(gt_dir, args.gt_dir)  # only choose thye just gt in the list
        os.makedirs(join(gt_dir ), exist_ok=True)
        sys.exit(f"** There is no ground truth for this dataset! in you {gt_dir} **")

    key = args.key  # x = scipy.io.loadmat(filename)[key]
    file_format = args.file_format  # ".mat", ".npy" or ".png"

    thrs = 99 if args.full else 9
    print("Edge-maps source: ", result_dir)
    if args.nms:
        print("Applying NMS?", args.nms)
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
