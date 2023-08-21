import datetime
import torch
import util.misc as misc


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = float(0.)
        self.avg = float(0.)
        self.sum = float(0.)
        self.count = float(0.)

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, loss_scaler, clip_max_norm, writer, args):
    total_steps = 0
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    device = next(model.parameters()).device

    if writer is not None:
        print('log_dir: {}'.format(writer.log_dir))

    optimizer.zero_grad()
    aux_optimizer.zero_grad()

    # calculate runtime
    t0 = datetime.datetime.now()

    for i, (samples, t_scores, s_scores) in enumerate(metric_logger.log_every(train_dataloader, print_freq, header)):
        total_steps += samples.shape[0]
        samples = samples.to(device)

        out_net = model(samples, t_scores, s_scores)

        out_criterion = criterion(out_net, samples)
        out_criterion['loss'] /= accum_iter
        aux_loss = model.aux_loss()
        aux_loss /= accum_iter
        if (i + 1) % accum_iter == 0:
            out_criterion["loss"].backward()
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()

            aux_loss.backward()
            aux_optimizer.step()

            optimizer.zero_grad()
            aux_optimizer.zero_grad()

        torch.cuda.synchronize()

        loss_value = out_criterion["loss"].item()
        L1_loss_value = out_criterion["L1_loss"].item()
        ssim_loss_value = out_criterion["ssim_loss"].item()
        vgg_loss_value = out_criterion["vgg_loss"].item()
        bpp_loss_value = out_criterion["bpp_loss"].item()
        aux_loss_value = aux_loss.item()

        metric_logger.update(loss=loss_value)
        metric_logger.update(L1_loss=L1_loss_value)
        metric_logger.update(ssim_loss=ssim_loss_value)
        metric_logger.update(vgg_loss=vgg_loss_value)
        metric_logger.update(bpp_loss=bpp_loss_value)
        metric_logger.update(aux_loss=aux_loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        L1_loss_value_reduce = misc.all_reduce_mean(L1_loss_value)
        ssim_loss_value_reduce = misc.all_reduce_mean(ssim_loss_value)
        vgg_loss_value_reduce = misc.all_reduce_mean(vgg_loss_value)
        bpp_loss_value_reduce = misc.all_reduce_mean(bpp_loss_value)
        aux_loss_value_reduce = misc.all_reduce_mean(aux_loss_value)

        if writer is not None and (i + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((i / len(train_dataloader) + epoch) * 1000)
            writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            writer.add_scalar('L1_loss', L1_loss_value_reduce, epoch_1000x)
            writer.add_scalar('ssim_loss', ssim_loss_value_reduce, epoch_1000x)
            writer.add_scalar('vgg_loss', vgg_loss_value_reduce, epoch_1000x)
            writer.add_scalar('bpp_loss', bpp_loss_value_reduce, epoch_1000x)
            writer.add_scalar('aux_loss', aux_loss_value_reduce, epoch_1000x)
            writer.add_scalar('lr', max_lr, epoch_1000x)

        if i % 500 == 0:
            t1 = datetime.datetime.now()
            deltatime = t1 - t0
            dt = deltatime.seconds + 1e-6 * deltatime.microseconds
            print(f"Train epoch {epoch}: ["
                  f"{i*len(samples)}/{len(train_dataloader.dataset)}"
                  f" ({100. * i / len(train_dataloader):.0f}%)]"
                  f'\tTime: {dt:.2f} |'
                  f'\tLoss: {out_criterion["loss"].item():.3f} |'
                  f'\tL1 loss: {out_criterion["L1_loss"].item():.3f} |'
                  f'\tSSIM loss: {out_criterion["ssim_loss"].item():.3f} |'
                  f'\tVgg loss: {out_criterion["vgg_loss"].item():.3f} |'
                  f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                  f"\tAux loss: {aux_loss.item():.2f}")
            t0 = t1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}


def test_epoch(epoch, test_dataloader, model, criterion):
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    L1_loss = AverageMeter()
    ssim_loss = AverageMeter()
    vgg_loss = AverageMeter()
    aux_loss = AverageMeter()

    device = next(model.parameters()).device

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for (samples, t_scores, s_scores) in metric_logger.log_every(test_dataloader, 10, header):
            samples = samples.to(device)
            with torch.cuda.amp.autocast():
                out_net = model(samples, t_scores, s_scores)
                # compute output
                out_criterion = criterion(out_net, samples)

                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                ssim_loss.update(out_criterion["ssim_loss"])
                vgg_loss.update(out_criterion["vgg_loss"])
                L1_loss.update(out_criterion["L1_loss"])

            metric_logger.update(loss=loss.avg)
            metric_logger.update(L1_loss=L1_loss.avg)
            metric_logger.update(ssim_loss=ssim_loss.avg)
            metric_logger.update(vgg_loss=vgg_loss.avg)
            metric_logger.update(bpp_loss=bpp_loss.avg)
            metric_logger.update(aux_loss=aux_loss.avg)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"Test epoch {epoch}: Average losses:"
          f"\tLoss: {loss.avg:.3f} |"
          f"\tL1 loss: {L1_loss.avg:.3f} |"
          f"\tSSIM loss: {ssim_loss.avg:.3f} |"
          f"\tVgg loss: {vgg_loss.avg:.3f} |"
          f"\tBpp loss: {bpp_loss.avg:.2f} |"
          f"\tAux loss: {aux_loss.avg:.2f}\n")

    return {k: round(meter.global_avg, 2) for k, meter in metric_logger.meters.items()}
