from copy import deepcopy
import os
import datetime
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from config import Config
from loss import PixLoss, ClsLoss
from dataset import MyData
from models.birefnet import BiRefNet, BiRefNetC2F
from utils import Logger, AverageMeter, set_seed, check_state_dict

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from torch.utils.tensorboard import SummaryWriter
from append import evaluate_evalset_by_cat, evaluate_testglass, MyLanceDataset
from tqdm import tqdm
import torch.distributed as dist


parser = argparse.ArgumentParser(description='')
parser.add_argument('--epochs', default=120, type=int)
parser.add_argument('--trainset', default='DIS5K', type=str, help="Options: 'DIS5K'")
parser.add_argument('--ckpt_dir', default='ckpt/tmp', help='Temporary folder')
parser.add_argument('--testsets', default='DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4', type=str)
parser.add_argument('--dist', default=False, type=lambda x: x == 'True')
parser.add_argument('--use_accelerate', action='store_true', help='`accelerate launch --multi_gpu train.py --use_accelerate`. Use accelerate for training, good for FP16/BF16/...')
parser.add_argument('--fasttest', default=False, type=lambda x: x == 'True')
parser.add_argument('--flexai', default=False, type=lambda x: x == 'True')
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--size', default=1024, type=int)
args = parser.parse_args()


if args.use_accelerate:
    from accelerate import Accelerator
    accelerator = Accelerator(
        mixed_precision=['no', 'fp16', 'bf16', 'fp8'][1],
        gradient_accumulation_steps=1,
    )
    args.dist = False


config = Config(flexai=args.flexai, batch_size=args.batch_size, size=args.size)
if config.rand_seed:
    set_seed(config.rand_seed)

# DDP
to_be_distributed = args.dist
print(f"to_be_distributed: {to_be_distributed}")
if to_be_distributed:
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*10))
    device = int(os.environ["LOCAL_RANK"])
else:
    device = config.device


main_gpu = 0
n_gpus = torch.cuda.device_count()
fl_main = device == main_gpu
print(f"device: {device}, main_gpu: {main_gpu}, fl_main: {fl_main}")


epoch_st = 1
# make dir for ckpt
print('flexai:', args.flexai)
if args.flexai:
    args.ckpt_dir = '/output/'
    config.weights['swin_v1_l'] = '/input/swin_large_patch4_window12_384_22kto1k.pth'


print(f'\n\nweights dir: {config.weights["swin_v1_l"]}\n\n')
os.makedirs(args.ckpt_dir, exist_ok=True)
# Init log file
logger = Logger(os.path.join(args.ckpt_dir, "log.txt"), fl_main=fl_main)
writer = SummaryWriter(log_dir=os.path.join(args.ckpt_dir, 'logs'))
logger_loss_idx = 1


lastpath = os.path.join(args.ckpt_dir, 'last.pth')
resume = False
if os.path.exists(lastpath):
    resume = True
logger.info("Resume: {}".format(resume))


# log model and optimizer params
# logger.info("Model details:"); logger.info(model)
if args.use_accelerate and accelerator.mixed_precision != 'no':
    config.compile = False


logger.info("datasets: load_all={}, compile={}.".format(config.load_all, config.compile))
logger.info("Other hyperparameters:")
logger.info(args)
logger.info(f'batch size: {config.batch_size}')
logger.info(f'fasttest: {args.fasttest}')
logger.info(f'fasttest: {args.fasttest}')


if os.path.exists(os.path.join(config.data_root_dir, config.task, args.testsets.strip('+').split('+')[0])):
    args.testsets = args.testsets.strip('+').split('+')
else:
    args.testsets = []


# Init model
def prepare_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, to_be_distributed=False, is_train=True):
    if to_be_distributed:
        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=min(config.num_workers, batch_size), pin_memory=True,
            shuffle=False, sampler=DistributedSampler(dataset), drop_last=True
        )
    else:
        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=min(config.num_workers, batch_size, 0), pin_memory=True,
            shuffle=is_train, drop_last=True
        )


def my_init_data_loaders(to_be_distributed):
    print('\n\n\n')
    import pathlib
    for p in pathlib.Path("/").rglob("*.lance"):
        print(p)

    train_fpath = '/input/2024_1101_sodlist_curated_512px_nobadhaginghands+notransparency.lance'
    logger.info("train_fpath: {}".format(train_fpath))
    train_ds = MyLanceDataset(train_fpath, long=config.size[0])

    train_loader = prepare_dataloader(
        train_ds,
        config.batch_size, to_be_distributed=to_be_distributed, is_train=True
    )
    test_loaders = {}
    return train_loader, test_loaders


def init_data_loaders(to_be_distributed):
    # Prepare dataset
    train_loader = prepare_dataloader(
        MyData(datasets=config.training_set, image_size=config.size, is_train=True),
        config.batch_size, to_be_distributed=to_be_distributed, is_train=True
    )
    print(len(train_loader), "batches of train dataloader {} have been created.".format(config.training_set))
    test_loaders = {}
    for testset in args.testsets:
        _data_loader_test = prepare_dataloader(
            MyData(datasets=testset, image_size=config.size, is_train=False),
            config.batch_size_valid, is_train=False
        )
        print(len(_data_loader_test), "batches of valid dataloader {} have been created.".format(testset))
        test_loaders[testset] = _data_loader_test
    return train_loader, test_loaders


def init_models_optimizers(epochs, to_be_distributed):
    if config.model == 'BiRefNet':
        print(f'\n\nweights dir: {config.weights["swin_v1_l"]}\n\n')
        model = BiRefNet(bb_pretrained=True and not resume, config=config)
    elif config.model == 'BiRefNetC2F':
        model = BiRefNetC2F(bb_pretrained=True and not resume)

    # args.resume = '/home/rafael/workspace/BiRefNet/BiRefNet-general-epoch_244.pth'
    # args.resume = '/home/rafael/workspace/BiRefNet/ckpt/y-curated-notransp/last.pth'
    print(f'\n\nweights dir: {config.weights["swin_v1_l"]}\n\n')
    if resume:
        logger.info("=> loading checkpoint '{}'".format(lastpath))
        states = torch.load(lastpath, map_location='cpu')
        state_dict = states['model_state']
        state_dict = check_state_dict(state_dict)
        model.load_state_dict(state_dict)

        ep_resume = states['steps_cur']
        global epoch_st
        epoch_st = ep_resume + 1

    if not args.use_accelerate:
        if to_be_distributed:
            model = model.to(device)
            model = DDP(model, device_ids=[device])
        else:
            model = model.to(device)

    if config.compile and not args.fasttest:
        model = torch.compile(model, mode=['default', 'reduce-overhead', 'max-autotune'][0])

    if config.precisionHigh:
        torch.set_float32_matmul_precision('high')


    # Setting optimizer
    if config.optimizer == 'AdamW':
        optimizer = optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=1e-2)
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[lde if lde > 0 else epochs + lde + 1 for lde in config.lr_decay_epochs],
        gamma=config.lr_decay_rate
    )
    logger.info("Optimizer details:"); logger.info(optimizer)
    logger.info("Scheduler details:"); logger.info(lr_scheduler)

    if resume:
        optimizer.load_state_dict(states['optimizer_state'])
        optimizer.param_groups[0]['lr'] = config.lr
        logger.info("Loaded optimizer checkpoint")
        sadlog_best = states['sadlog_best']
        logger.info("Loaded sadlog_best: {:.2f}".format(sadlog_best))
        sad_glass = states['sad_glass']
        logger.info("Loaded sad_glass: {:.2f}".format(sad_glass))
    else:
        sadlog_best = 9999
        sad_glass = 9999

    return model, optimizer, lr_scheduler, sadlog_best, sad_glass


class Trainer:
    def __init__(
        self, data_loaders, model_opt_lrsch,
    ):
        self.model, self.optimizer, self.lr_scheduler, self.sadlog_best, self.sad_glass = model_opt_lrsch

        self.train_loader, self.test_loaders = data_loaders

        if args.use_accelerate:
            self.train_loader, self.model, self.optimizer = accelerator.prepare(self.train_loader, self.model, self.optimizer)
            for testset in self.test_loaders.keys():
                self.test_loaders[testset] = accelerator.prepare(self.test_loaders[testset])
        if config.out_ref:
            self.criterion_gdt = nn.BCELoss()

        # Setting Losses
        self.pix_loss = PixLoss()
        self.cls_loss = ClsLoss()

        # Others
        self.loss_log = AverageMeter()
        if config.lambda_adv_g:
            self.optimizer_d, self.lr_scheduler_d, self.disc, self.adv_criterion = self._load_adv_components()
            self.disc_update_for_odd = 0

    def _load_adv_components(self):
        # AIL
        from loss import Discriminator
        disc = Discriminator(channels=3, img_size=config.size)
        if to_be_distributed:
            disc = disc.to(device)
            disc = DDP(disc, device_ids=[device], broadcast_buffers=False)
        else:
            disc = disc.to(device)
        if config.compile:
            disc = torch.compile(disc, mode=['default', 'reduce-overhead', 'max-autotune'][0])
        adv_criterion = nn.BCELoss()
        if config.optimizer == 'AdamW':
            optimizer_d = optim.AdamW(params=disc.parameters(), lr=config.lr, weight_decay=1e-2)
        elif config.optimizer == 'Adam':
            optimizer_d = optim.Adam(params=disc.parameters(), lr=config.lr, weight_decay=0)
        lr_scheduler_d = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_d,
            milestones=[lde if lde > 0 else args.epochs + lde + 1 for lde in config.lr_decay_epochs],
            gamma=config.lr_decay_rate
        )
        return optimizer_d, lr_scheduler_d, disc, adv_criterion

    def _train_batch(self, batch):
        if args.use_accelerate:
            inputs = batch[0]#.to(device)
            gts = batch[1]#.to(device)
            class_labels = batch[2]#.to(device)
        else:
            inputs = batch[0].to(device)
            gts = batch[1].to(device)
            class_labels = batch[2].to(device)

        scaled_preds, class_preds_lst = self.model(inputs)
        if config.out_ref:
            (outs_gdt_pred, outs_gdt_label), scaled_preds = scaled_preds
            for _idx, (_gdt_pred, _gdt_label) in enumerate(zip(outs_gdt_pred, outs_gdt_label)):
                _gdt_pred = nn.functional.interpolate(_gdt_pred, size=_gdt_label.shape[2:], mode='bilinear', align_corners=True).sigmoid()
                _gdt_label = _gdt_label.sigmoid()
                loss_gdt = self.criterion_gdt(_gdt_pred, _gdt_label) if _idx == 0 else self.criterion_gdt(_gdt_pred, _gdt_label) + loss_gdt
            # self.loss_dict['loss_gdt'] = loss_gdt.item()
        if None in class_preds_lst:
            loss_cls = 0.
        else:
            loss_cls = self.cls_loss(class_preds_lst, class_labels) * 1.0
            self.loss_dict['loss_cls'] = loss_cls.item()

        # Loss
        loss_pix = self.pix_loss(scaled_preds, torch.clamp(gts, 0, 1)) * 1.0
        self.loss_dict['loss_pix'] = loss_pix.item()
        # since there may be several losses for sal, the lambdas for them (lambdas_pix) are inside the loss.py
        loss = loss_pix + loss_cls
        if config.out_ref:
            loss = loss + loss_gdt * 1.0

        if config.lambda_adv_g:
            # gen
            valid = Variable(torch.cuda.FloatTensor(scaled_preds[-1].shape[0], 1).fill_(1.0), requires_grad=False).to(device)
            adv_loss_g = self.adv_criterion(self.disc(scaled_preds[-1] * inputs), valid) * config.lambda_adv_g
            loss += adv_loss_g
            self.loss_dict['loss_adv'] = adv_loss_g.item()
            self.disc_update_for_odd += 1
        self.loss_log.update(loss.item(), inputs.size(0))
        self.optimizer.zero_grad()
        if args.use_accelerate:
            accelerator.backward(loss)
        else:
            loss.backward()
        self.optimizer.step()

        if config.lambda_adv_g and self.disc_update_for_odd % 2 == 0:
            # disc
            fake = Variable(torch.cuda.FloatTensor(scaled_preds[-1].shape[0], 1).fill_(0.0), requires_grad=False).to(device)
            adv_loss_real = self.adv_criterion(self.disc(gts * inputs), valid)
            adv_loss_fake = self.adv_criterion(self.disc(scaled_preds[-1].detach() * inputs.detach()), fake)
            adv_loss_d = (adv_loss_real + adv_loss_fake) / 2 * config.lambda_adv_d
            self.loss_dict['loss_adv_d'] = adv_loss_d.item()
            self.optimizer_d.zero_grad()
            adv_loss_d.backward()
            self.optimizer_d.step()

    def save_ckpt(self, path, epoch):
        state = self.model.module.state_dict() if \
                to_be_distributed or args.use_accelerate else \
                self.model.state_dict()

        torch.save({
            'steps_cur': epoch,
            'model_state': deepcopy(state),
            'optimizer_state': deepcopy(self.optimizer.state_dict()),
            'sadlog_best': self.sadlog_best,
            'sad_glass': self.sad_glass,
        }, path)
        logger.info('Model saved at {}'.format(path))

    def train_epoch(self, epoch):
        global logger_loss_idx
        self.model.train()
        self.loss_dict = {}
        # if True:
        if epoch > args.epochs + config.finetune_last_epochs:
            if config.task == 'Matting':
                self.pix_loss.lambdas_pix_last['mae'] *= 1
                self.pix_loss.lambdas_pix_last['mse'] *= 0.9
                self.pix_loss.lambdas_pix_last['ssim'] *= 0.9
            else:
                self.pix_loss.lambdas_pix_last['bce'] *= 0
                self.pix_loss.lambdas_pix_last['ssim'] *= 1
                self.pix_loss.lambdas_pix_last['iou'] *= 0.5
                self.pix_loss.lambdas_pix_last['mae'] *= 0.9

        pbar_trainloader = tqdm(
            self.train_loader, desc=f"ep {epoch}", disable=device != 0)

        # for batch_idx, batch in enumerate(self.train_loader):
        for batch_idx, batch in enumerate(pbar_trainloader):
            self._train_batch(batch)
            # Logger
            if batch_idx % 30 == 0:
                #info_progress = 'Epoch[{0}/{1}] Iter[{2}/{3}].'.format(epoch, args.epochs, batch_idx, len(self.train_loader))
                #info_loss = 'Training Losses'
                info_loss = ''
                for loss_name, loss_value in self.loss_dict.items():
                    info_loss += ', {}: {:.3f}'.format(loss_name, loss_value)
                #logger.info(' '.join((info_progress, info_loss)))

                pbar_trainloader.set_description(info_loss)

                if args.fasttest and batch_idx > 0:
                    break

        info_loss = '@==Final== Epoch[{0}/{1}]  Training Loss: {loss.avg:.3f}  '.format(epoch, args.epochs, loss=self.loss_log)
        logger.info(info_loss)
        if device == main_gpu:
            writer.add_scalars('raw_objective', {'train': self.loss_log.avg}, epoch)

        if device == main_gpu:
            self.model.eval()

            sadlog = evaluate_evalset_by_cat(
                    self.model, fl_fasttest=args.fasttest, long=config.size[0],
                    flexai=args.flexai
                    )
            logger.info('Epoch[{0}/{1}]  sadlog: {sadlog:.2f}'.format(epoch, args.epochs, sadlog=sadlog))
            writer.add_scalars('sad-log', {'testcat': sadlog}, epoch)

            # sadglass = evaluate_testglass(self.model, fl_fasttest=args.fasttest)
            # logger.info('Epoch[{0}/{1}]  sad-glass3: {sadglass:.2f}'.format(epoch, args.epochs, sadglass=sadglass))
            # writer.add_scalars('sad', {'test-glass-3': sadglass}, epoch)

            if sadlog < self.sadlog_best:
                modelpath = os.path.join(args.ckpt_dir, 'sadlog_best.pth'.format(epoch))
                self.sadlog_best = sadlog
                self.save_ckpt(modelpath, epoch)

            #if sadglass < self.sad_glass:
            #    modelpath = os.path.join(args.ckpt_dir, 'sadglass_best.pth'.format(epoch))
            #    self.sad_glass = sadglass
            #    self.save_ckpt(modelpath, epoch)

            self.save_ckpt(lastpath, epoch)

        self.model.train()


        self.lr_scheduler.step()
        if config.lambda_adv_g:
            self.lr_scheduler_d.step()
        return self.loss_log.avg


def main():

    if args.flexai:
        init_function = my_init_data_loaders
    else:
        init_function = init_data_loaders

    trainer = Trainer(
            data_loaders=init_function(to_be_distributed),
            model_opt_lrsch=init_models_optimizers(args.epochs, to_be_distributed)
    )

    for epoch in range(epoch_st, args.epochs+1):
        train_loss = trainer.train_epoch(epoch)
        # Save checkpoint
        # DDP
        # if epoch >= args.epochs - config.save_last and epoch % config.save_step == 0:
        # if True:
        #     modelpath = os.path.join(args.ckpt_dir, 'last.pth'.format(epoch))
        #     torch.save(
        #         trainer.model.module.state_dict() if to_be_distributed or args.use_accelerate else trainer.model.state_dict(),
        #         # os.path.join(args.ckpt_dir, 'epoch_{}.pth'.format(epoch))
        #         modelpath
        #     )
        #     logger.info('Model saved at {}'.format(modelpath))

    if to_be_distributed:
        destroy_process_group()

if __name__ == '__main__':
    main()
