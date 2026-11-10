# -*- coding: utf-8 -*-
"""只扩展 Stage 2 的 ViewBridge Trainer。

原始 ``clustercontrast/trainers.py`` 保持不变。这里复制 Stage 2 Trainer 的
训练顺序，并只在第一段原始损失之后附加 ViewBridge；第二段联合 memory
训练 ``loss2``、EMA 更新和所有原 ReID 损失均保持原顺序。
"""

from __future__ import absolute_import, print_function

import time

import torch

from .utils.meters import AverageMeter
from .viewbridge import (add_view_bridge_loss, bridge_is_enabled,
                         compute_view_bridge_loss)


class ClusterContrastTrainer_PCLMP_ViewBridge(object):
    """带可关闭 ViewBridge 的 Stage 2 Trainer。"""

    def __init__(self, encoder, encoder_ema, memory=None, view_score_map=None,
                 lambda_view_bridge=0.0, view_weight_mode="none"):
        super(ClusterContrastTrainer_PCLMP_ViewBridge, self).__init__()
        self.encoder = encoder
        self.encoder_ema = encoder_ema
        self.memory_ir = memory
        self.memory_rgb = memory
        self.memory_all = memory

        # view_score_map 的键就是原 Preprocessor 返回的 fname，值中至少包含
        # bridge_weight。None 表示没有显式提供分数文件，必须退化为 baseline。
        self.view_score_map = view_score_map
        self.lambda_view_bridge = float(lambda_view_bridge)
        self.view_weight_mode = view_weight_mode

    def train(self, epoch, data_loader_ir, data_loader_rgb,
              data_loader_all_ir, data_loader_all_rgb, optimizer,
              print_freq=10, train_iters=400, i2r=None, r2i=None):
        self.encoder.train()
        self.encoder_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        original_losses = AverageMeter()
        bridge_losses = AverageMeter()
        weighted_bridge_losses = AverageMeter()

        # 每个 epoch 汇总，而不是每个 iteration 大量刷屏。
        epoch_stats = {
            "valid_samples": 0,
            "positive_weight_samples": 0,
            "missing_score": 0,
            "missing_bgm": 0,
            "invalid_label": 0,
            "invalid_target": 0,
            "non_finite": 0,
            "weight_sum": 0.0,
        }
        enabled = bridge_is_enabled(
            self.lambda_view_bridge, self.view_weight_mode, self.view_score_map)

        end = time.time()
        for i in range(train_iters):
            # 原空中 batch 已经包含 fnames：
            # (imgs, fnames, pseudo_labels, camera_ids, indexes)。
            inputs_ir_raw = data_loader_ir.next()
            inputs_rgb_raw = data_loader_rgb.next()
            data_time.update(time.time() - end)

            inputs_ir, fnames_ir, labels_ir, indexes_ir = self._parse_data_ir_with_names(inputs_ir_raw)
            inputs_rgb, inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb_raw)

            inputs_rgb = torch.cat((inputs_rgb, inputs_rgb1), 0)
            labels_rgb = torch.cat((labels_rgb, labels_rgb), -1)
            _, f_out_rgb, f_out_ir, labels_rgb, labels_ir, pool_rgb, pool_ir = self._forward(
                inputs_rgb, inputs_ir, label_1=labels_rgb, label_2=labels_ir, modal=0)

            loss_ir = self.memory_ir(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)

            # 原始 cross contrastive learning：i2r 是空中簇到地面簇，
            # r2i 是地面簇到空中簇。本段保持原实现和交替策略。
            if r2i:
                rgb2ir_labels = torch.tensor(
                    [r2i[key.item()] for key in labels_rgb], device=labels_rgb.device)
                ir2rgb_labels = torch.tensor(
                    [i2r[key.item()] for key in labels_ir], device=labels_ir.device)
                if epoch % 2 == 1:
                    cross_loss = self.memory_rgb(f_out_ir, ir2rgb_labels.long())
                else:
                    cross_loss = self.memory_ir(f_out_rgb, rgb2ir_labels.long())
            else:
                # 与原来的标量 0 等价，但放在正确设备上，便于空映射安全运行。
                cross_loss = f_out_ir.new_zeros(())

            with torch.no_grad():
                _, f_out_rgb_ema, f_out_ir_ema, labels_rgb_ema, labels_ir_ema, \
                    pool_rgb_ema, pool_ir_ema = self._forward_ema(
                        inputs_rgb, inputs_ir, label_1=labels_rgb,
                        label_2=labels_ir, modal=0)
            loss_ir_ema = self.memory_ir(f_out_ir_ema, labels_ir_ema, model_name="encoder_ema")
            loss_rgb_ema = self.memory_rgb(f_out_rgb_ema, labels_rgb_ema, model_name="encoder_ema")
            loss_ema = loss_ir_ema + loss_rgb_ema

            # 这是原 Stage 2 第一段完整损失，先原样计算，再决定是否附加 Bridge。
            original_loss = loss_ir + loss_rgb + 0.25 * cross_loss + loss_ema
            if enabled:
                bridge_loss, batch_stats = compute_view_bridge_loss(
                    aerial_features=f_out_ir,
                    aerial_labels=labels_ir,
                    image_paths=list(fnames_ir),
                    ground_prototypes=self.memory_rgb.features,
                    bgm_i2r=i2r,
                    view_score_map=self.view_score_map,
                    # CMhard 的 features 有 2*N 行；只有前 N 行是合法均值原型索引。
                    ground_num_clusters=self.memory_rgb.num_samples)
                loss = add_view_bridge_loss(
                    original_loss, bridge_loss, self.lambda_view_bridge,
                    self.view_weight_mode)
                for key in ("valid_samples", "positive_weight_samples", "missing_score",
                            "missing_bgm", "invalid_label", "invalid_target", "non_finite"):
                    epoch_stats[key] += batch_stats[key]
                epoch_stats["weight_sum"] += batch_stats["weight_sum"]
            else:
                # 关闭时不查分数、不查 BGM，也不改变原始计算图。
                bridge_loss = original_loss.detach().new_zeros(())
                loss = original_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            original_losses.update(original_loss.item())
            bridge_losses.update(bridge_loss.item())
            weighted_bridge_losses.update(self.lambda_view_bridge * bridge_loss.item())

            # 第二段联合模态训练完全沿用原逻辑，Bridge 不参与。
            inputs_all_ir = data_loader_all_ir.next()
            inputs_all_rgb = data_loader_all_rgb.next()
            inputs_all_ir, labels_all_ir, indexes_all_ir = self._parse_data_ir(inputs_all_ir)
            inputs_all_rgb, inputs_all_rgb1, labels_all_rgb, indexes_all_rgb = self._parse_data_rgb(inputs_all_rgb)
            inputs_all_rgb = torch.cat((inputs_all_rgb, inputs_all_rgb1), 0)
            labels_all_rgb = torch.cat((labels_all_rgb, labels_all_rgb), -1)

            _, f_out_all_rgb, f_out_all_ir, labels_all_rgb, labels_all_ir, \
                pool_all_rgb, pool_all_ir = self._forward(
                    inputs_all_rgb, inputs_all_ir,
                    label_1=labels_all_rgb, label_2=labels_all_ir, modal=0)
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
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Original {:.3f} ({:.3f})\t'
                      'Bridge {:.6f}\t'
                      'Weighted Bridge {:.6f}\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      'Loss ema {:.3f}\t'
                      'Loss ir ema {:.3f}\t'
                      'Loss rgb ema {:.3f}\t'
                      'Loss all {:.3f}\t'
                      'Loss all ir {:.3f}\t'
                      'Loss all rgb {:.3f}\t'.format(
                          epoch, i + 1, len(data_loader_rgb),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg,
                          losses.val, losses.avg,
                          original_losses.val, original_losses.avg,
                          bridge_losses.val, weighted_bridge_losses.val,
                          loss_ir, loss_rgb, loss_ema, loss_ir_ema, loss_rgb_ema,
                          loss2, loss_all_ir, loss_all_rgb))

        valid = epoch_stats["valid_samples"]
        mean_weight = epoch_stats["weight_sum"] / float(valid) if valid else 0.0
        summary = dict(epoch_stats)
        summary.update({
            "enabled": enabled,
            "mean_weight": mean_weight,
            "original_loss": original_losses.avg,
            "bridge_loss": bridge_losses.avg,
            "weighted_bridge_loss": weighted_bridge_losses.avg,
            "total_loss": losses.avg,
        })
        print('ViewBridge:\n'
              '  enabled={}\n'
              '  original_loss={:.6f}\n'
              '  valid_samples={}\n'
              '  positive_weight_samples={}\n'
              '  missing_score={}\n'
              '  missing_bgm={}\n'
              '  invalid_label={}\n'
              '  invalid_target={}\n'
              '  non_finite={}\n'
              '  mean_weight={:.6f}\n'
              '  bridge_loss={:.6f}\n'
              '  weighted_bridge_loss={:.6f}'.format(
                  enabled, summary["original_loss"], summary["valid_samples"],
                  summary["positive_weight_samples"], summary["missing_score"],
                  summary["missing_bgm"], summary["invalid_label"],
                  summary["invalid_target"], summary["non_finite"],
                  summary["mean_weight"], summary["bridge_loss"],
                  summary["weighted_bridge_loss"]))
        return summary

    def _parse_data_rgb(self, inputs):
        imgs, imgs1, _, pids, _, indexes = inputs
        return imgs.cuda(), imgs1.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir_with_names(self, inputs):
        imgs, fnames, pids, _, indexes = inputs
        return imgs.cuda(), fnames, pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, label_1=None, label_2=None, modal=0):
        return self.encoder(x1, x2, modal=modal, label_1=label_1, label_2=label_2)

    def _forward_ema(self, x1, x2, label_1=None, label_2=None, modal=0):
        return self.encoder_ema(x1, x2, modal=modal, label_1=label_1, label_2=label_2)

    def _update_ema_variables(self, model, ema_model, alpha):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

