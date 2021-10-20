import torch

from trainer.vanilla import Vanilla
from utils.utils import accuracy, AverageMeter, save_checkpoint
from kd_losses import SoftTarget
from models import model_dict


class SemiOnlineKnowledgeDistillation(Vanilla):
    def __init__(self, params):
        # Model
        self.teacher_name = params.get('teacher_name')
        self.teacher_depth = params.get('teacher_depth', '')
        self.teacher_widen = params.get('teacher_widen', '')
        self.teacher_checkpoint = params.get('teacher_checkpoint')
        self.teacher = None
        # Coefficient
        self.lambda_kd = params.get('lambda_kd', 1)
        self.lambda_ce = params.get('lambda_ce', 1)
        self.auxiliary_lambda_kd_t = params.get('auxiliary_lambda_kd_t', 1)
        self.auxiliary_lambda_kd_s = params.get('auxiliary_lambda_kd_s', 1)
        self.auxiliary_lambda_ce = params.get('auxiliary_lambda_ce', 1)
        self.lr_auxiliary = params.get('lr_auxiliary', 0.05)

        self.distillation_name = params.get('distillation_name', 'soft_target')
        self.criterion_kd = SoftTarget(T=4)
        self.auxiliary_index = -3
        self.best_top1_A = 0
        experimental_name = f"Teacher-{self.teacher_name}{self.teacher_depth}-{self.teacher_widen}"

        super().__init__(params, experimental_name)

    def run(self):
        self.set_data()
        self.set_model()
        self.load_teacher()
        self.set_optimizer_scheduler()
        self.train_model()

    def load_teacher(self):
        if self.teacher_name == 'wideresnet':
            self.teacher = model_dict[f"wrn_{self.teacher_depth}_{self.teacher_widen}"](
                num_classes=self.num_classes)
        else:
            assert False, f'Not considering {self.teacher_name}'
        if torch.cuda.device_count() > 1:
            self.teacher = torch.nn.DataParallel(self.teacher)
        self.teacher = self.teacher.to(self.device)
        if self.teacher_checkpoint:
            state = torch.load(self.teacher_checkpoint)['model']
            teacher_state_dict = self.teacher.state_dict()
            loaded_state = {k: v for k, v in state.items() if k in teacher_state_dict}
            teacher_state_dict.update(loaded_state)
            self.teacher.load_state_dict(teacher_state_dict)
            self.logger.info("Load teacher's checkpoint done!")
        else:
            self.logger.info("No teacher's checkpoint!")
        top1, _ = self.evaluation_vanilla(self.teacher)
        self.logger.info(f'Teacher ACC: {top1.avg}')
        for k, v in self.teacher.named_parameters():
            if 'auxiliary' not in k:
                v.requires_grad = False

    def train(self):
        self.model.train()
        self.teacher.train()
        # log of student
        total_loss = AverageMeter()
        total_loss_ce = AverageMeter()
        total_loss_kd = AverageMeter()
        total_top1 = AverageMeter()
        total_top5 = AverageMeter()
        # log of auxiliary
        total_loss_A = AverageMeter()
        total_loss_ce_A = AverageMeter()
        total_loss_kd_T_A = AverageMeter()
        total_loss_kd_S_A = AverageMeter()
        total_top1_A = AverageMeter()
        total_top5_A = AverageMeter()

        for batch_id, (data, targets) in enumerate(self.train_loader):
            data = data.to(self.device)
            targets = targets.to(self.device)

            feature_S, output_S = self.model(data, is_feat=True)
            feature_T, output_T = self.teacher(data, is_feat=True)
            feature_A, output_A = self.teacher.auxiliary_forward(feature_T[self.auxiliary_index].detach())
            # loss of auxiliary
            loss_kd_T_A, loss_kd_S_A, loss_kd = self.calculate_kd(self.distillation_name, feature_S, feature_A,
                                                                  feature_T, output_S, output_A, output_T)
            loss_ce_A = self.criterion_ce(output_A, targets) * self.auxiliary_lambda_ce
            loss_A = loss_ce_A + loss_kd_T_A + loss_kd_S_A
            # loss of student
            loss_ce = self.criterion_ce(output_S, targets) * self.lambda_ce
            loss = loss_ce + loss_kd
            loss_total = loss_A + loss
            # accuracy
            top1, top5 = accuracy(output_S, targets, topk=(1, 5))
            top1_A, top5_A = accuracy(output_A, targets, topk=(1, 5))
            # update parameter of student
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            # update log of student
            total_loss.update(loss.item(), data.size(0))
            total_loss_ce.update(loss_ce.item(), data.size(0))
            total_loss_kd.update(loss_kd.item(), data.size(0))
            total_top1.update(top1.item(), data.size(0))
            total_top5.update(top5.item(), data.size(0))
            # update log of auxiliary
            total_loss_A.update(loss_A.item(), data.size(0))
            total_loss_ce_A.update(loss_ce_A.item(), data.size(0))
            total_loss_kd_T_A.update(loss_kd_T_A.item(), data.size(0))
            total_loss_kd_S_A.update(loss_kd_S_A.item(), data.size(0))
            total_top1_A.update(top1_A.item(), data.size(0))
            total_top5_A.update(top5_A.item(), data.size(0))

        info_str = f"Train (Branch)=> loss_ce: {total_loss_ce_A.avg:.4f}, loss_kd_T_A: {total_loss_kd_T_A.avg:.4f}," \
                   f"loss_kd_S_A: {total_loss_kd_S_A.avg:.4f}, prec@1: {total_top1_A.avg:.2f}, prec@5: {total_top5_A.avg:.2f}"
        self.logger.info(info_str)

        info_str = f"Train (Student)=> loss_ce: {total_loss_ce.avg:.4f}, loss_kd: {total_loss_kd.avg:.4f}, " \
                   f"prec@1: {total_top1.avg:.2f}, prec@5: {total_top5.avg:.2f}"
        self.logger.info(info_str)

        return total_top1, total_top5

    @torch.no_grad()
    def evaluation(self, model):
        model.eval()
        self.teacher.eval()
        total_top1 = AverageMeter()
        total_top5 = AverageMeter()
        total_top1_t = AverageMeter()
        total_top5_t = AverageMeter()
        for batch_id, (data, targets) in enumerate(self.test_loader):
            data = data.to(self.device)
            targets = targets.to(self.device)
            output_S = model(data)
            feature_T, output_T = self.teacher(data, is_feat=True)
            _, output_A = self.teacher.auxiliary_forward(feature_T[self.auxiliary_index].detach())
            top1, top5 = accuracy(output_S, targets, topk=(1, 5))
            total_top1.update(top1.item(), data.size(0))
            total_top5.update(top5.item(), data.size(0))
            top1_t, top5_t = accuracy(output_A, targets, topk=(1, 5))
            total_top1_t.update(top1_t.item(), data.size(0))
            total_top5_t.update(top5_t.item(), data.size(0))
        if total_top1_t.avg > self.best_top1_A:
            self.best_top1_A = total_top1_t.avg
            state_dict = {'model': self.teacher.state_dict()}
            save_checkpoint(state_dict, True, f"{self.save_folder}/teacher")
        self.logger.info(
            f"Test (branch)=> lr:{self.optimizer.param_groups[1]['lr']}, "
            f"top1_A:{total_top1_t.avg:.2f}, top5_A:{total_top5_t.avg:.2f}, @Best: {self.best_top1_A}")

        return total_top1, total_top5


    def set_optimizer_scheduler(self):
        self.optimizer = torch.optim.SGD([{'params': self.model.parameters()},
                                          {'params': self.teacher.parameters(), 'lr': self.lr_auxiliary}],
                                         lr=self.lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.milestones)

    def calculate_kd(self, name, feature_S, feature_A, feature_T, output_S, output_A, output_T):
        if name == 'soft_target':
            loss_kd_T_A = self.criterion_kd(output_A, output_T.detach()) * self.auxiliary_lambda_kd_t
            loss_kd_S_A = self.criterion_kd(output_A, output_S.detach()) * self.auxiliary_lambda_kd_s
            loss_S = self.criterion_kd(output_S, output_A.detach()) * self.lambda_kd
        else:
            assert NotImplementedError, f"No considering {name}"

        return loss_kd_T_A, loss_kd_S_A, loss_S
