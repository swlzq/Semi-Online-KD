import logging
import colorlog
import os
import time
import shutil
import torch
import random
import numpy as np
from shutil import copyfile


def create_logger():
    """
        Setup the logging environment
    """
    log = logging.getLogger()  # root logger
    log.setLevel(logging.DEBUG)
    format_str = '%(asctime)s - %(levelname)-8s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    if os.isatty(2):
        cformat = '%(log_color)s' + format_str
        colors = {'DEBUG': 'reset',
                  'INFO': 'reset',
                  'WARNING': 'bold_yellow',
                  'ERROR': 'bold_red',
                  'CRITICAL': 'bold_red'}
        formatter = colorlog.ColoredFormatter(cformat, date_format,
                                              log_colors=colors)
    else:
        formatter = logging.Formatter(format_str, date_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    return logging.getLogger(__name__)


class TimeRecorder(object):
    """
    Recode training time.
    """

    def __init__(self, start_epoch, epochs, logger):
        self.total_time = 0.
        self.remaining_time = 0.
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.logger = logger
        self.start_time = time.time()

    def update(self):
        now_time = time.time()
        elapsed_time = now_time - self.start_time
        self.start_time = now_time
        self.total_time += elapsed_time
        self.remaining_time = elapsed_time * (self.epochs - self.start_epoch)
        self.start_epoch += 1

        self.logger.info(f'Cost time=>{self.format_time(self.total_time)}')
        self.logger.info(f'Remaining time=>{self.format_time(self.remaining_time)}')

    @staticmethod
    def format_time(time):
        h = time // 3600
        m = (time % 3600) // 60
        s = (time % 3600) % 60
        return f'{h}h{m}m{s:.2f}s'


def output_process(output_path):
    if os.path.exists(output_path):
        print("{} file exist!".format(output_path))
        action = input("Select Action: d (delete) / q (quit):").lower().strip()
        act = action
        if act == 'd':
            shutil.rmtree(output_path)
        else:
            raise OSError("Directory {} exits!".format(output_path))

    if not os.path.exists(output_path):
        os.makedirs(output_path)


def save_code(src, dst, exclude=[]):
    """
    Save experimental codes.
    """
    for f in os.listdir(src):
        # Do not save experimental results
        if f in exclude:
            continue
        src_file = os.path.join(src, f)
        file_split = f.split(".")
        if len(file_split) >= 2:
            if not os.path.isdir(dst):
                os.makedirs(dst)
            dst_file = os.path.join(dst, f)
            try:
                shutil.copyfile(src=src_file, dst=dst_file)
            except:
                print("Copy file error! src: {}, dst: {}".format(src_file, dst_file))
        elif os.path.isdir(src_file):
            deeper_dst = os.path.join(dst, f)
            save_code(src_file, deeper_dst)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
        # res.append(correct_k)
    return res


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_opts(opts, save_path='.'):
    with open(f"{save_path}/opts.txt", 'w') as f:
        for k, v in opts.items():
            f.write(str(k) + ": " + str(v) + '\n')


def save_checkpoint(state_dict, is_best, folder_name='.'):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    checkpoint_name = f"{folder_name}/checkpoint.pth.tar"
    torch.save(state_dict, checkpoint_name)
    if is_best:
        model_name = f"{folder_name}/best_model.pth.tar"
        copyfile(checkpoint_name, model_name)


def fix_random(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return True


def count_parameters_in_MB(model):
    return sum(np.prod(v.size()) for name, v in model.named_parameters()) / 1e6
