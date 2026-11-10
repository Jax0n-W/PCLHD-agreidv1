"""Stage 2 trainer with confidence-gated partial cross-view supervision."""
from __future__ import absolute_import, print_function

import time

import torch

from .confident_alignment import alignment_weight_at_epoch
from .trainers import ClusterContrastTrainer_PCLMP
from .utils.meters import AverageMeter


def masked_cross_memory_loss(features, labels, mapping, memory):
    """Apply a cross-memory target only to labels present in ``mapping``."""
    source_labels = labels.detach().cpu().tolist()
    selected = [index for index, label in enumerate(source_labels)
                if int(label) in mapping]
    if not selected:
        return features.new_zeros(()), 0, len(source_labels)
    index = torch.tensor(selected, dtype=torch.long, device=features.device)
    targets = torch.tensor(
        [mapping[int(source_labels[item])] for item in selected],
        dtype=torch.long, device=features.device)
    loss = memory(features.index_select(0, index), targets)
    return loss, len(selected), len(source_labels)


class ClusterContrastTrainer_PCLMP_ConfidentAlignment(
        ClusterContrastTrainer_PCLMP):
    """Preserve Stage 2 while replacing forced BGM with reliable partial BGM."""

    def __init__(self, encoder, encoder_ema, alignment_weight=0.25,
                 warmup_epochs=5, ramp_epochs=10, include_ema_loss=True):
        super(ClusterContrastTrainer_PCLMP_ConfidentAlignment, self).__init__(
            encoder, encoder_ema)
        self.alignment_weight = alignment_weight
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.include_ema_loss = include_ema_loss

    def train(self, epoch, data_loader_ir, data_loader_rgb,
              data_loader_all_ir, data_loader_all_rgb, optimizer,
              print_freq=10, train_iters=400, i2r=None, r2i=None):
        self.encoder.train()
        self.encoder_ema.train()
        i2r = i2r or {}
        r2i = r2i or {}
        cross_weight = alignment_weight_at_epoch(
            epoch, self.alignment_weight, self.warmup_epochs,
            self.ramp_epochs)
        print("[Confident Alignment] epoch={} weight={:.4f} pairs={}".format(
            epoch, cross_weight, len(r2i)))

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        matched = 0
        eligible = 0
        end = time.time()

        for i in range(train_iters):
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)
            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb, inputs_rgb1, labels_rgb, indexes_rgb = \
                self._parse_data_rgb(inputs_rgb)
            inputs_rgb = torch.cat((inputs_rgb, inputs_rgb1), 0)
            labels_rgb = torch.cat((labels_rgb, labels_rgb), -1)
            (_, f_out_rgb, f_out_ir, labels_rgb, labels_ir, pool_rgb,
             pool_ir) = self._forward(
                inputs_rgb, inputs_ir, label_1=labels_rgb,
                label_2=labels_ir, modal=0)

            loss_ir = self.memory_ir(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            # ClusterMemory mutates its prototypes in autograd backward even
            # when multiplied by zero. During warmup, skip the forward
            # entirely so the cross branch has genuinely no side effects.
            if cross_weight == 0.0:
                cross_loss = f_out_ir.new_zeros(())
                batch_matched = 0
                batch_eligible = int(labels_ir.numel())
            elif epoch % 2 == 1:
                cross_loss, batch_matched, batch_eligible = \
                    masked_cross_memory_loss(
                        f_out_ir, labels_ir, i2r, self.memory_rgb)
            else:
                cross_loss, batch_matched, batch_eligible = \
                    masked_cross_memory_loss(
                        f_out_rgb, labels_rgb, r2i, self.memory_ir)
            matched += batch_matched
            eligible += batch_eligible

            with torch.no_grad():
                (_, f_out_rgb_ema, f_out_ir_ema, labels_rgb_ema,
                 labels_ir_ema, pool_rgb_ema, pool_ir_ema) = \
                    self._forward_ema(
                        inputs_rgb, inputs_ir, label_1=labels_rgb,
                        label_2=labels_ir, modal=0)
            loss_ir_ema = self.memory_ir(
                f_out_ir_ema, labels_ir_ema, model_name="encoder_ema")
            loss_rgb_ema = self.memory_rgb(
                f_out_rgb_ema, labels_rgb_ema, model_name="encoder_ema")
            loss_ema = loss_ir_ema + loss_rgb_ema
            loss = loss_ir + loss_rgb + cross_weight * cross_loss
            if self.include_ema_loss:
                loss = loss + loss_ema

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item())

            inputs_all_ir = data_loader_all_ir.next()
            inputs_all_rgb = data_loader_all_rgb.next()
            inputs_all_ir, labels_all_ir, indexes_all_ir = \
                self._parse_data_ir(inputs_all_ir)
            inputs_all_rgb, inputs_all_rgb1, labels_all_rgb, indexes_all_rgb = \
                self._parse_data_rgb(inputs_all_rgb)
            inputs_all_rgb = torch.cat((inputs_all_rgb, inputs_all_rgb1), 0)
            labels_all_rgb = torch.cat(
                (labels_all_rgb, labels_all_rgb), -1)
            (_, f_out_all_rgb, f_out_all_ir, labels_all_rgb, labels_all_ir,
             pool_all_rgb, pool_all_ir) = self._forward(
                inputs_all_rgb, inputs_all_ir, label_1=labels_all_rgb,
                label_2=labels_all_ir, modal=0)
            loss_all_ir = self.memory_all(f_out_all_ir, labels_all_ir)
            loss_all_rgb = self.memory_all(f_out_all_rgb, labels_all_rgb)
            loss2 = loss_all_ir + loss_all_rgb
            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()
            self._update_ema_variables(self.encoder, self.encoder_ema, 0.999)

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % print_freq == 0:
                coverage = matched / float(eligible) if eligible else 0.0
                print(
                    "Epoch: [{}][{}/{}]\tTime {:.3f} ({:.3f})\t"
                    "Data {:.3f} ({:.3f})\tLoss {:.3f} ({:.3f})\t"
                    "Loss ir {:.3f}\tLoss rgb {:.3f}\tLoss cross {:.3f}\t"
                    "Cross weight {:.4f}\tMatched {:.1%}\tLoss ema {:.3f}\t"
                    "Loss all {:.3f}".format(
                        epoch, i + 1, len(data_loader_rgb), batch_time.val,
                        batch_time.avg, data_time.val, data_time.avg,
                        losses.val, losses.avg, loss_ir, loss_rgb, cross_loss,
                        cross_weight, coverage, loss_ema, loss2))

