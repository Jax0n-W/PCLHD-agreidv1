from __future__ import absolute_import, print_function

import time

import torch

from .trainers import ClusterContrastTrainer_PCLMP
from .utils.meters import AverageMeter


def compose_stage2_first_loss(loss_ir, loss_rgb, cross_loss, loss_ema,
                              include_ema_loss=True):
    """Compose Stage 2's first loss for the two No-ALL ablations."""
    loss = loss_ir + loss_rgb + 0.25 * cross_loss
    if include_ema_loss:
        loss = loss + loss_ema
    return loss


class ClusterContrastTrainer_PCLMP_NoALL(ClusterContrastTrainer_PCLMP):
    """Remove only Stage 2's second ALL-memory optimization.

    The first optimization retains ``loss_ema``. EMA forward, EMA parameter
    averaging, clustering, BGM, and EMA evaluation remain controlled by the
    original training pipeline.
    """

    include_ema_loss = True
    ablation_name = "KEEP EMA LOSS + NO ALL LOSS"

    def train(self, epoch, data_loader_ir, data_loader_rgb, data_loader_all_ir,
              data_loader_all_rgb, optimizer, print_freq=10, train_iters=400,
              i2r=None, r2i=None):
        self.encoder.train()
        self.encoder_ema.train()

        print("============================================")
        print("ABLATION: {}".format(self.ablation_name))
        print("Trainer source: {}".format(__file__))
        print("EMA loss included: {}".format(self.include_ema_loss))
        print("ALL memory optimization enabled: False")
        print("============================================")

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

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
            _, f_out_rgb, f_out_ir, labels_rgb, labels_ir, pool_rgb, pool_ir = \
                self._forward(inputs_rgb, inputs_ir, label_1=labels_rgb,
                              label_2=labels_ir, modal=0)

            loss_ir = self.memory_ir(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)

            if r2i:
                rgb2ir_labels = torch.tensor(
                    [r2i[key.item()] for key in labels_rgb]).cuda()
                ir2rgb_labels = torch.tensor(
                    [i2r[key.item()] for key in labels_ir]).cuda()
                alternate = True
                if alternate:
                    if epoch % 2 == 1:
                        cross_loss = self.memory_rgb(
                            f_out_ir, ir2rgb_labels.long())
                    else:
                        cross_loss = self.memory_ir(
                            f_out_rgb, rgb2ir_labels.long())
                else:
                    cross_loss = (
                        self.memory_rgb(f_out_ir, ir2rgb_labels.long())
                        + self.memory_ir(f_out_rgb, rgb2ir_labels.long()))
            else:
                cross_loss = loss_ir.new_zeros(())

            with torch.no_grad():
                (_, f_out_rgb_ema, f_out_ir_ema, labels_rgb_ema,
                 labels_ir_ema, pool_rgb_ema, pool_ir_ema) = self._forward_ema(
                    inputs_rgb, inputs_ir, label_1=labels_rgb,
                    label_2=labels_ir, modal=0)
            loss_ir_ema = self.memory_ir(
                f_out_ir_ema, labels_ir_ema, model_name="encoder_ema")
            loss_rgb_ema = self.memory_rgb(
                f_out_rgb_ema, labels_rgb_ema, model_name="encoder_ema")
            loss_ema = loss_ir_ema + loss_rgb_ema

            loss = compose_stage2_first_loss(
                loss_ir, loss_rgb, cross_loss, loss_ema,
                include_ema_loss=self.include_ema_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item())

            # The original second ALL-memory forward/backward/optimizer step is
            # intentionally disabled. Keep explicit zero monitor values so the
            # original-style log cannot reference undefined local variables.
            loss2 = loss.detach().new_zeros(())
            loss_all_ir = loss.detach().new_zeros(())
            loss_all_rgb = loss.detach().new_zeros(())

            self._update_ema_variables(self.encoder, self.encoder_ema, 0.999)

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      'Loss ema {:.3f}\t'
                      'Loss ir ema {:.3f}\t'
                      'Loss rgb ema {:.3f}\t'
                      'Loss all DISABLED {:.3f}\t'
                      'Loss all ir DISABLED {:.3f}\t'
                      'Loss all rgb DISABLED {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg, loss_ir, loss_rgb,
                              loss_ema, loss_ir_ema, loss_rgb_ema, loss2,
                              loss_all_ir, loss_all_rgb))
