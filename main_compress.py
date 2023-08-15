import argparse
import math
import random
import shutil
import os
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from util.pos_embed import interpolate_pos_embed
from compressai.zoo import models

from torch.utils.tensorboard import SummaryWriter

import util.misc as misc
import numpy as np

from engine_compress import train_one_epoch, test_epoch

from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.dataset import CreateDatasetFromImages


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"].values())
        out["ssim_loss"] = output["loss"][0]
        out["L1_loss"] = output["loss"][1]
        out["vgg_loss"] = output["loss"][2]

        # gather
        out["loss"] = self.lmbda * (0.25 * out["ssim_loss"] + 10 * out["L1_loss"] + 0.1 * out["vgg_loss"]) + out["bpp_loss"]
        return out

def configure_optimizers(net, args):
    """Return two optimizers"""
    parameters = {n for n, p in net.named_parameters() if not n.endswith(".quantiles") and p.requires_grad}
    aux_parameters = {n for n, p in net.named_parameters() if n.endswith(".quantiles") and p.requires_grad}

    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters

    assert len(inter_params) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer

def save_checkpoint(state, is_best, filename):
    filename1 = filename[:-8] + '_epoch' + str(state['epoch']) + filename[-8:]
    torch.save(state, filename1)
    if is_best:
        shutil.copyfile(filename1, filename[:-8] + "_best" + filename[-8:])

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument(
        "-m",
        "--model",
        default="MCM",
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Training dataset path")
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
    )
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-4,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: %(default)s)")
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=8,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-4,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--save", action="store_true", default=True, help="Save model to disk")
    parser.add_argument("--save_path", type=str, default="ckpt/model.pth.tar", help="Where to Save model")
    parser.add_argument("--seed", default=0, type=int, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, default='', help="Path to a checkpoint")
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--vis_num', type=int, default=64, help='number of visual patches to input the model')
    return parser

def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    train_dataset = CreateDatasetFromImages(True, args.dataset)
    test_dataset = CreateDatasetFromImages(False, args.dataset)

    if True:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        print("Sampler_train = %s" % str(sampler_train))
        sampler_val = torch.utils.data.SequentialSampler(test_dataset)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(test_dataset)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.log_dir)
    else:
        writer = None

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    test_dataloader = DataLoader(test_dataset, sampler=sampler_val, batch_size=args.test_batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)

    net = models[args.model](visual_tokens=args.vis_num)
    '''load mae encoder model'''
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.checkpoint)
        checkpoint_model = checkpoint['model']
        state_dict = net.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        # interpolate position embedding
        interpolate_pos_embed(net, checkpoint_model)
        msg = net.load_state_dict(checkpoint_model)

    if args.resume:
        checkpoint_model_mae = torch.load("./checkpoint/checkpoint-79.pth", map_location='cpu')["model"]
        # checkpoint = torch.load(args.resume, map_location='cpu')
        # msg = net.load_state_dict(checkpoint["model"])
        for k, v in net.named_parameters():
            if k in list(checkpoint_model_mae.keys()) and not k.startswith('decoder'):
                v.requires_grad = False

    net.to(device)
    names = []
    for name, param in net.named_parameters():
        if param.requires_grad:
            names.append(name)
    # print("Param groups = %s" % json.dumps(names, indent=2))
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    optimizer, aux_optimizer = configure_optimizers(net, args)
    # print("lr: %f" % optimizer.param_groups[0]['lr'])
    # print("effective batch size: %d" % eff_batch_size)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=4)
    loss_scaler = NativeScaler()
    misc.load_model(args=args, model=net, optimizer=optimizer, aux_optimizer=aux_optimizer, loss_scaler=loss_scaler)

    criterion = RateDistortionLoss(lmbda=args.lmbda)

    print(f"Start training for {args.epochs} epochs")
    last_epoch = 0 + args.start_epoch
    optimizer.param_groups[0]['lr'] = args.learning_rate
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_stats = train_one_epoch(net, criterion, train_dataloader, optimizer, aux_optimizer, epoch, loss_scaler, args.clip_max_norm, writer=writer, args=args)

        test_stats, lpips_test = test_epoch(epoch, test_dataloader, net, criterion)
        loss = test_stats['loss']

        if args.output_dir:
            misc.save_model(args=args, epoch=epoch, model=net, optimizer=optimizer, aux_optimizer=aux_optimizer, loss_scaler=loss_scaler)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, **{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch, 'n_parameters': n_parameters}
        # if args.output_dir and misc.is_main_process():
        #     if writer is not None:
        #         writer.flush()
        #     with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        #         f.write(json.dumps(log_stats) + "\n")
        #     log_lpips = {'epoch': epoch,'lpips':lpips_test.item(),'bpp loss':test_stats['bpp_loss']}
        #     with open(os.path.join(args.output_dir, "lpips.txt"), mode="a", encoding="utf-8") as f:
        #         f.write(json.dumps(log_lpips) + "\n")

        # loss = test_epoch(epoch, test_dataloader, net, criterion)
        # lr_scheduler.step(loss)

        # is_best = loss < best_loss
        # best_loss = min(loss, best_loss)

        # if args.save:
        #     save_checkpoint(
        #         {
        #             "epoch": epoch,
        #             "state_dict": net.state_dict(),
        #             "loss": loss,
        #             "optimizer": optimizer.state_dict(),
        #             "aux_optimizer": aux_optimizer.state_dict(),
        #             "lr_scheduler": lr_scheduler.state_dict(),
        #         },
        #         is_best,
        #         args.save_path,
        #         # args.save_path[:-8]+'_epoch'+str(epoch)+args.save_path[-8:],
        #     )

        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #                 **{f'test_{k}': v for k, v in test_stats.items()},
        #                 'epoch': epoch,
        #                 'n_parameters': n_parameters}

        # if args.output_dir and misc.is_main_process():
        #     if log_writer is not None:
        #         log_writer.flush()
        #     with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        #         f.write(json.dumps(log_stats) + "\n")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
