import os, sys
import time
from datetime import datetime
import argparse
from eval_component import eval_one_epoch
num_test = {"BSDS": 200, "BRIND": 200, "NYUD": 654, "BIPED": 50, "UDED": 30, "TEEDedges":30}


def need_test(root, dset, full, file_format, is_nms=True):
    flag = False
    sub_pth=os.listdir(root)
    tmp_dirs = ['nms', 'png']
    if (tmp_dirs[0] in sub_pth and tmp_dirs[0] in sub_pth):
        sub_pth = tmp_dirs# os.listdir(root)
    flag_dir = "nms-eval" if full else "nms-eval-9"

    format_dir_map = {
        ".mat": "mat",
        ".npy": "npy",
        ".png": "png"
    }

    format_dir = format_dir_map.get(file_format)
    if format_dir is None:
        raise ValueError(f"Formato no soportado: {file_format}")

    format_path = os.path.join(root, format_dir) if is_nms else \
        os.path.join(root, "nms")
    if not os.path.exists(format_path) and not os.path.split(root)[-1]==format_dir:
        print(f"** There is not {format_path} folder, we've created for you, restart!**")
        os.makedirs(format_path)
        sys.exit()
    if format_dir in sub_pth and os.path.isdir(format_path) and flag_dir not in sub_pth:
        file_num = len(os.listdir(format_path))
        if file_num == num_test[dset]:
            flag = True

    return flag



def getFlist(args):
    dirs_ = set()
    tmp_res_dir = os.path.join("result",args.eval_dir)
    if not os.path.exists(tmp_res_dir):
        print(f"** There is not {tmp_res_dir} folder, we created for you, restart!**")
        os.makedirs(tmp_res_dir)
        sys.exit()

    file_dirs = tmp_res_dir.split(' ')
    print("*" * 40)
    print("file dirs> ", file_dirs)

    for file_dir in file_dirs:
        #for root, _, _ in os.walk(file_dir):
        if need_test(file_dir, args.dataset, args.full, args.file_format, args.nms):
            dirs_.add(file_dir)
            print(f"Working on dir: {file_dir}")
    print("*" * 40 + "\n")
    return dirs_


class Quit_timer(object):
    def __init__(self, T, limit):
        self.time_flag = 0
        self.sleep_time = T
        self.quit_ceiling = limit

    def sleep(self):
        if self.time_flag == 0:
            print(datetime.now().strftime("%Y-%m-%d %H:%M"), end=' ')
            print("no available results, start to sleep ...")

        else:
            print(datetime.now().strftime("%Y-%m-%d %H:%M"), end=' ')
            print("no avaliable png:")
        time.sleep(self.sleep_time * 3600)
        self.time_flag += self.sleep_time
        self.quit()

    def quit(self):
        if self.time_flag >= self.quit_ceiling:
            print("No new data has been generated for {} hours, the program automatically exits".format(
                self.quit_ceiling))
            exit(1)

    def refresh(self):
        self.time_flag = 0


class Parser_One_Epoch(object):
    def __init__(self, root, dataset, full, file_format, is_nms, gt_dir):
        self.root = root
        self.dataset = dataset
        self.key = "groundTruth"
        self.file_format = file_format
        self.full = full
        self.nms = is_nms
        self.gt_dir = gt_dir

def main_func(args):
    qtimer = Quit_timer(args.T, args.limit)
    print("*" * 40)
    print(args)
    print("*" * 40 + "\n")
    cast =True
    while cast:
        fset = getFlist(args)
        print(fset)
        print(f" len fset {len(fset)}")
        if len(fset) != 0:
            for updataset in fset:
                parser_one_epoch = Parser_One_Epoch(updataset, args.dataset, args.full,
                                                     args.file_format, args.nms, args.gt_dir)
                mes = eval_one_epoch(parser_one_epoch)

            qtimer.refresh()
            cast = False if mes else True
        else:
            if args.notwait:
                print("no avaliable result and args.notwait is true, return process")
                return
            qtimer.sleep()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval output')
    parser.add_argument('--T', type=float, default=0.5, help="sleep time,defult 0.5 hour")
    parser.add_argument('--limit', type=float, default=8, help="time of empty cycle(hours)")
    parser.add_argument("-nw", '--notwait', action="store_true", help="whether wait new result")
    parser.add_argument("-d", '--dataset', default="UDED")
    parser.add_argument("-m", '--model', default="MatchED")
    parser.add_argument("-f", '--full', action="store_true")
    parser.add_argument("-nms", '--nms', action="store_true")
    parser.add_argument("-gt", '--gt_dir', type=str, default="gt_ed",
                        help="For UDED, the gt folder may be gt_ed or gt_bd")

    parser.add_argument('--file_format', type=str, default=".png", choices=[".mat", ".npy", ".png"],
                    help="File formats")

    parser.add_argument("-ed",'--eval_dir',
                        default=parser.parse_args().model+"-"+parser.parse_args().dataset)
    args = parser.parse_args()

    if args.dataset is not None:
        for dataset in num_test.keys():
            if dataset in args.dataset.upper():
                args.dataset = dataset
                break
    else:
        for dataset in num_test.keys():
            if dataset in args.eval_dir.upper():
                args.dataset = dataset
                break

    if args.dataset is None:
        raise Exception("Point out dataset in test dir OR point out dataset in args.dataset")
    print(f"Starting to eval result on dataset: {args.dataset}, using format: {args.file_format} on predicted images")
    main_func(args)