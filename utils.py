import torch

import csv
import shutil
import pathlib
from copy import deepcopy
from os import remove
from os.path import isfile
from collections import OrderedDict


def load_model(model, ckpt_file, main_gpu, use_cuda: bool=True, strict=True):
    r"""Load model for training, resume training, evaluation,
    quantization and finding similar kernels for new methods
    """
    if use_cuda:
        checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage.cuda(main_gpu))
        try:
            model.load_state_dict(checkpoint, strict)
        except:
            model.module.load_state_dict(checkpoint, strict)
    else:
        checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
        try:
            model.load_state_dict(checkpoint)
        except:
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                if k[:7] == 'module.':
                    name = k[7:] # remove `module.`
                else:
                    name = k[:]
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)

    return checkpoint


def save_model(arch_name, dataset, state, ckpt_name='ckpt_best.pth'):
    r"""Save the model (checkpoint) at the training time
    """
    dir_ckpt = pathlib.Path('checkpoint')
    dir_path = dir_ckpt / arch_name / dataset
    dir_path.mkdir(parents=True, exist_ok=True)

    if ckpt_name is None:
        ckpt_name = 'ckpt_best.pth'
    model_file = dir_path / ckpt_name
    torch.save(state, model_file)


def save_summary(arch_name, dataset, name, summary):
    r"""Save summary i.e. top-1/5 validation accuracy in each epoch
    under `summary` directory
    """
    dir_summary = pathlib.Path('summary')
    dir_path = dir_summary / 'csv'
    dir_path.mkdir(parents=True, exist_ok=True)

    file_name = '{}_{}_{}.csv'.format(arch_name, dataset, name)
    file_summ = dir_path / file_name

    if summary[0] == 0:
        with open(file_summ, 'w', newline='') as csv_out:
            writer = csv.writer(csv_out)
            header_list = ['Epoch', 'Acc@1_train', 'Acc@5_train', 'Acc@1_valid', 'Acc@5_valid']
            writer.writerow(header_list)
            writer.writerow(summary)
    else:
        file_temp = dir_path / 'temp.csv'
        shutil.copyfile(file_summ, file_temp)
        with open(file_temp, 'r', newline='') as csv_in:
            with open(file_summ, 'w', newline='') as csv_out:
                reader = csv.reader(csv_in)
                writer = csv.writer(csv_out)
                for row_list in reader:
                    writer.writerow(row_list)
                writer.writerow(summary)
        remove(file_temp)


def save_eval(summary):
    r"""Save evaluation results i.e. top-1/5 test accuracy in the `eval.csv` file
    """
    dir_summary = pathlib.Path('summary')
    dir_path = dir_summary / 'csv'
    dir_path.mkdir(parents=True, exist_ok=True)

    file_summ = dir_path / 'eval.csv'
    if not isfile(file_summ):
        with open(file_summ, 'w', newline='') as csv_out:
            writer = csv.writer(csv_out)
            header_list = ['ckpt', 'Acc@1', 'Acc@5']
            writer.writerow(header_list)
            writer.writerow(summary)
    else:
        file_temp = 'temp.csv'
        shutil.copyfile(file_summ, file_temp)
        with open(file_temp, 'r', newline='') as csv_in:
            with open(file_summ, 'w', newline='') as csv_out:
                reader = csv.reader(csv_in)
                writer = csv.writer(csv_out)
                for row_list in reader:
                    writer.writerow(row_list)
                writer.writerow(summary)
        remove(file_temp)


class AverageMeter(object):
    r"""Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class ScoreMeter(object):
    r"""Stores the ground truth and prediction labels
    to compute the f1-score (macro)
    """
    def __init__(self):
        self.label = []
        self.prediction = []
        self.score = None

    def update(self, output, target):
        pred = torch.argmax(output, dim=-1)
        self.prediction += pred.detach().cpu().tolist()
        self.label += target.detach().cpu().tolist()


def set_scheduler(optimizer, args):
    r"""Sets the learning rate scheduler
    """
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, args.gamma)
    elif args.scheduler == 'multistep':
        if int(args.warmup_lr_epoch) > 0:
            scheduler = GradualWarmupScheduler(optimizer, args.warmup_lr_epoch, args.milestones, args.lr, args.gamma, args.epochs, args.warmup_lr)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, args.gamma)
    elif args.scheduler == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.step_size)
    else:
        print('==> unavailable scheduler!! exit...\n')
        exit()

    return scheduler


def accuracy(output, target, topk=(1,)):
    r"""Computes the accuracy over the $k$ top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def set_arch_name(args):
    r"""Set architecture name
    """
    arch_name = deepcopy(args.arch)
    if args.arch in ['resnet']:
        arch_name += str(args.layers)
    elif args.arch in ['wideresnet']:
        arch_name += '{}_{}'.format(args.layers, int(args.width_mult))
    
    return arch_name


class GradualWarmupScheduler(object):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, warmup_epoch, milestones, init_lr, gamma, total_epoch, warmup_init_lr=0.1):
        self.optimizer = optimizer
        self.milestones = [0, warmup_epoch] + milestones + [total_epoch]
        self.interval_epoch = [(self.milestones[i], self.milestones[i + 1]) for i in range(len(self.milestones) - 1)]
        self.cur_interval_idx = 0
        self.interval_lr = [(warmup_init_lr, init_lr)]
        for i in range(len(milestones) + 1):
            self.interval_lr.append(init_lr * gamma ** i)
        self.total_epoch = total_epoch
        self.finished = False
        self.lr = warmup_init_lr if warmup_epoch > 0 else init_lr
        super(GradualWarmupScheduler, self).__init__()

    def get_lr(self):
        return self.lr

    def step(self, epoch=None, metrics=None):
        lr_change = 0
        if epoch >= self.interval_epoch[self.cur_interval_idx][1]:
            self.cur_interval_idx += 1
            self.lr = self.interval_lr[self.cur_interval_idx]
            lr_change = 1
        if self.cur_interval_idx == 0:
            start_lr, end_lr = self.interval_lr[0]
            start_epoch, end_epoch = self.interval_epoch[0]
            self.lr = start_lr + (end_lr - start_lr) * (epoch - start_epoch) / (end_epoch - start_epoch)
            lr_change = 1
        if lr_change:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

