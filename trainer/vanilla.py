import torch.nn as nn
import torch
from tqdm import tqdm

from trainer.base_trainer import BaseTrainer
from models import model_dict
from utils.utils import count_parameters_in_MB, AverageMeter, accuracy, save_checkpoint
from dataset import get_dataloader


class Vanilla(BaseTrainer):
    def __init__(self, params, experimental_name=''):
        # Data
        self.data_name = params.get('data_name')
        self.data_path = params.get('data_path')
        self.num_classes = params.get('num_classes', 100)
        self.train_loader = None
        self.test_loader = None
        # Model
        self.model_name = params.get('model_name')
        self.model_depth = params.get('model_depth', '')
        self.model_widen = params.get('model_widen', '')
        self.model_checkpoint = params.get('model_checkpoint')
        self.model = None
        self.testing = params.get('evaluation', False)
        # Base training settings
        self.start_epoch = params.get('start_epoch', 1)
        self.epochs = params.get('epochs', 200)
        self.batch_size = params.get('batch_size', 128)
        self.lr = params.get('lr', 0.1)
        self.device = params.get('device', 'cuda')
        self.milestones = params.get('milestones', [200])
        self.optimizer = None
        self.scheduler = None
        self.criterion_ce = nn.CrossEntropyLoss()
        # Log
        self.best_top1 = 0
        self.best_top5 = 0
        self.best_epoch = 0

        seed = params.get('seed', None)
        experimental_name = f"{self.__class__.__name__}_{self.model_name}{self.model_depth}-{self.model_widen}_{self.data_name}_" \
                            f"{experimental_name}/{params.get('name', 'debug')}"
        super().__init__(experimental_name, seed)

    def run(self):
        self.set_data()
        self.set_model()
        self.set_optimizer_scheduler()
        self.train_model()

    def train_model(self):
        if self.model_checkpoint:
            state_dict = torch.load(self.model_checkpoint)
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.scheduler.load_state_dict(state_dict['scheduler'])
            self.best_top1 = state_dict['best_top1']
            self.best_top5 = state_dict['best_top5']
            self.best_epoch = state_dict['best_epoch']
            self.start_epoch = state_dict['start_epoch']
            self.logger.info("Load model's checkpoint done!")
        if self.testing:
            self.logger.info("Start testing model...")
            top1, top5 = self.evaluation_vanilla(self.model)
            self.logger.info(f"top1:{top1.avg:.2f}, top5:{top5.avg:.2f}")
        else:
            self.logger.info("Start training model...")
            for epoch in tqdm(range(self.start_epoch, self.epochs + 1)):
                self.logger.info(f'Epoch[{epoch}/{self.epochs}]')
                self.train()
                top1, top5 = self.evaluation(self.model)
                self.writer.add_scalar('test/top1', top1.avg, epoch)
                is_best = False
                if top1.avg > self.best_top1:
                    self.best_top1 = top1.avg
                    self.best_top5 = top5.avg
                    self.best_epoch = epoch
                    is_best = True
                state_dict = {'model': self.model.state_dict(),
                              'optimizer': self.optimizer.state_dict(),
                              'scheduler': self.scheduler.state_dict(),
                              'best_top1': self.best_top1,
                              'best_top5': self.best_top5,
                              'best_epoch': self.best_epoch,
                              'start_epoch': epoch
                              }
                save_checkpoint(state_dict, is_best, f"{self.save_folder}/model")
                self.logger.info(
                    f"Test=> lr:{self.optimizer.param_groups[0]['lr']}, "
                    f"top1:{top1.avg:.2f}, top5:{top5.avg:.2f} "
                    f"@Best:({self.best_top1}, {self.best_top5}, {self.best_epoch})")
                self.scheduler.step()

    def train(self):
        self.model.train()
        total_loss = AverageMeter()
        total_top1 = AverageMeter()
        total_top5 = AverageMeter()
        for batch_id, (data, targets) in enumerate(self.train_loader):
            data = data.to(self.device)
            targets = targets.to(self.device)
            output = self.model(data)
            loss = self.criterion_ce(output, targets)
            top1, top5 = accuracy(output, targets, topk=(1, 5))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss.update(loss.item(), data.size(0))
            total_top1.update(top1.item(), data.size(0))
            total_top5.update(top5.item(), data.size(0))

        info_str = f"Train=> total_loss: {total_loss.avg}, " \
                   f"prec@1: {total_top1.avg}, prec@5: {total_top5.avg}"
        self.logger.info(info_str)

    @torch.no_grad()
    def evaluation_vanilla(self, model):
        model.eval()
        total_top1 = AverageMeter()
        total_top5 = AverageMeter()
        for batch_id, (data, targets) in enumerate(self.test_loader):
            data = data.to(self.device)
            targets = targets.to(self.device)
            output_S = model(data)
            top1, top5 = accuracy(output_S, targets, topk=(1, 5))
            total_top1.update(top1.item(), data.size(0))
            total_top5.update(top5.item(), data.size(0))

        return total_top1, total_top5
        
    @torch.no_grad()
    def evaluation(self, model):
        model.eval()
        total_top1 = AverageMeter()
        total_top5 = AverageMeter()
        for batch_id, (data, targets) in enumerate(self.test_loader):
            data = data.to(self.device)
            targets = targets.to(self.device)
            output_S = model(data)
            top1, top5 = accuracy(output_S, targets, topk=(1, 5))
            total_top1.update(top1.item(), data.size(0))
            total_top5.update(top5.item(), data.size(0))

        return total_top1, total_top5

    def set_data(self):
        self.train_loader, self.test_loader = get_dataloader(self.data_name, self.data_path, self.batch_size)

    def set_model(self):
        if self.data_name.startswith('CIFAR'):
            if self.model_name == 'wideresnet':
                self.model = model_dict[f"wrn_{self.model_depth}_{self.model_widen}"](num_classes=self.num_classes)
            else:
                assert False, f'Not considering {self.model_name}'
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.to(self.device)
        else:
            assert False, f"Not considering {self.data_name}"

    def set_optimizer_scheduler(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.milestones)
