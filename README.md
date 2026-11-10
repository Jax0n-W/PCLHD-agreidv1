# PCLHD AG-ReID.v1: D-AGVA × Three-Domain Ablation Release

本发布版用于在 **两个训练阶段均关闭通道转换增强** 的条件下，对
D-AGVA 与三域匹配进行受控的 2×2 消融实验。代码只面向 AG-ReID.v1。

## 固定实验矩阵

| 实验名 | Stage 1 D-AGVA | Stage 2 三域匹配 | 两阶段通道增强 |
|---|---:|---:|---:|
| `baseline` | 关闭 | 关闭 | 关闭 |
| `dynamic` | 开启 | 关闭 | 关闭 |
| `three_domain` | 关闭 | 开启 | 关闭 |
| `full` | 开启 | 开启 | 关闭 |

其中，通道增强关闭是指训练代码不调用 `ChannelAdapGray`、
`ChannelExchange` 或 `Gray`。`ChannelRandomErasing` 属于空间随机擦除，
四组实验均保留。D-AGVA 只作用于 Stage 1 的 aerial training branch；
Stage 2 始终使用原始图像。三域匹配只作用于 Stage 2。

四组实验还统一使用以下设置：

- Stage 2 prototype memory：`CMhard`；
- Stage 2 的 `loss_ema` 不参与反向传播；
- EMA encoder 参数更新保留；
- ALL-memory 第二次优化保留；
- 默认使用 Stage 1 最后一个 epoch 的 `checkpoint.pth.tar` 初始化 Stage 2；
- 不依据正式测试集 Rank-1/mAP 选择 checkpoint。

## 数据目录

默认训练根目录为：

```text
/home/lab338/Jaxon/dataset/AGreid_train
├── aerial_modify/1/bounding_box_train/
└── ground_modify/1/bounding_box_train/
```

D-AGVA 还需要与 aerial images 一一对应的 SCHP 前景掩码，默认位置：

```text
/home/lab338/Jaxon/dataset/AGreid_train_SCHP/aerial_modify/1/bounding_box_train/
```

三域匹配使用发布包中的：

```text
view_scores/agreid_aerial_view_scores_foreground.csv
```

数据集、掩码、模型权重和训练日志均未包含在发布包中。

## 环境

推荐 Python 3.10。安装依赖：

```bash
pip install -r requirements.txt
```

## 先做配置检查

下面的命令不会启动训练，只打印四组实验实际调用的子命令：

```bash
python train_agreid.py --experiment baseline --run-mode full --dry-run
python train_agreid.py --experiment dynamic --run-mode full --dry-run
python train_agreid.py --experiment three_domain --run-mode full --dry-run
python train_agreid.py --experiment full --run-mode full --dry-run
```

输出中应明确显示：

```text
ChannelAdapGray: False (Stage 1 and Stage 2)
ChannelExchange/Gray: False (Stage 1 and Stage 2)
Stage 2 memory: CMhard
Stage 2 EMA loss included: False
```

## 批量运行四组实验

批量脚本只训练两份 Stage 1（D=0、D=1），再分别启动四个 Stage 2，
避免重复训练相同的 Stage 1。

```bash
cd /path/to/PCLHD_AGReID_DT_release_20261110
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python scripts/run_all_ablations.py \
  --gpu 0 \
  --experiments baseline dynamic three_domain full \
  --data-dir /home/lab338/Jaxon/dataset/AGreid_train \
  --logs-dir /home/lab338/Jaxon/PCLHD_AGReID_DT_release/logs_agreid_dt \
  --mask-dir /home/lab338/Jaxon/dataset/AGreid_train_SCHP/aerial_modify/1/bounding_box_train \
  --aerial-image-root /home/lab338/Jaxon/dataset/AGreid_train/aerial_modify/1/bounding_box_train \
  --dynamic-agva-strict-masks \
  --epochs 50 --iters 400 --seed 1 --workers 8 --print-freq 50
```

建议先在同一命令末尾添加 `--dry-run` 核查路径。

## 单独运行

完整方法（D=1, T=1）：

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=0 \
python train_agreid.py \
  --experiment full --run-mode full \
  --data-dir /home/lab338/Jaxon/dataset/AGreid_train \
  --logs-dir /home/lab338/Jaxon/PCLHD_AGReID_DT_release/logs_agreid_dt \
  --dynamic-agva-mask-dir /home/lab338/Jaxon/dataset/AGreid_train_SCHP/aerial_modify/1/bounding_box_train \
  --dynamic-agva-image-root /home/lab338/Jaxon/dataset/AGreid_train/aerial_modify/1/bounding_box_train \
  --dynamic-agva-strict-masks \
  --epochs 50 --iters 400 --seed 1 --workers 8 --print-freq 50
```

只运行 Stage 2 时，必须显式给出 Stage 1 checkpoint：

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=0 \
python train_agreid.py \
  --experiment three_domain --run-mode stage2 \
  --stage1-checkpoint /path/to/stage1/checkpoint.pth.tar \
  --data-dir /home/lab338/Jaxon/dataset/AGreid_train \
  --logs-dir /home/lab338/Jaxon/PCLHD_AGReID_DT_release/logs_agreid_dt
```

## 日志与权重命名

Stage 1：

```text
logs_agreid_dt/agreid_dt_s1_d0_channel_off_seed1/1/checkpoint.pth.tar
logs_agreid_dt/agreid_dt_s1_d1_channel_off_seed1/1/checkpoint.pth.tar
```

Stage 2：

```text
logs_agreid_dt/agreid_dt_baseline_d0_t0_channel_off_s2_seed1/1/checkpoint.pth.tar
logs_agreid_dt/agreid_dt_dynamic_d1_t0_channel_off_s2_seed1/1/checkpoint.pth.tar
logs_agreid_dt/agreid_dt_three_domain_d0_t1_channel_off_s2_seed1/1/checkpoint.pth.tar
logs_agreid_dt/agreid_dt_full_d1_t1_channel_off_s2_seed1/1/checkpoint.pth.tar
```

`model_best.pth.tar` 仍可能由旧训练引擎写出，兼容历史结果；正式受控实验默认
不使用它。只有显式设置 `--checkpoint-policy legacy-best` 时，完整流程才会
用该文件初始化 Stage 2。

## 批量测试

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python scripts/test_all_checkpoints.py \
  --gpu 0 \
  --test-data-dir /home/lab338/Jaxon/dataset/AGreid_test \
  --logs-dir /home/lab338/Jaxon/PCLHD_AGReID_DT_release/logs_agreid_dt \
  --experiments baseline dynamic three_domain full
```

每个测试日志开头都会记录 D/T 开关、通道增强状态和 checkpoint 路径。
测试程序会输出 A$\rightarrow$G、G$\rightarrow$A，并保留现有脚本中的
G$\rightarrow$A+G 结果。论文主表应仅采用预先声明的正式评价协议；
G$\rightarrow$A+G 若非官方协议，应标为自定义补充实验。

## 本地验证

```bash
python -m unittest discover -s tests -v
python tools/verify_release.py
```

验证包括 2×2 开关、D-AGVA 仅在 Stage 1、三域匹配仅在 Stage 2、
CMhard/no-EMA-loss 固定设置、通道转换调用清除，以及发布包中不存在权重、
checkpoint、日志和 Python cache。

## 主要入口

- `train_agreid.py`：单组实验统一入口；
- `experiment_runner.py`：固定实验矩阵与子命令构造；
- `_train_agreid_engine.py`：Stage 1 与原始双域 Stage 2；
- `_train_agreid_three_domain_engine.py`：三域 Stage 2；
- `scripts/run_all_ablations.py`：四组消融批量训练；
- `scripts/test_all_checkpoints.py`：四组权重批量测试；
- `tests/test_release_configuration.py`：配置回归测试。
