from __future__ import absolute_import, print_function

from .trainers_no_all import ClusterContrastTrainer_PCLMP_NoALL


class ClusterContrastTrainer_PCLMP_NoEMA_NoALL(
        ClusterContrastTrainer_PCLMP_NoALL):
    """Remove ``loss_ema`` and the second ALL-memory optimization."""

    include_ema_loss = False
    ablation_name = "NO EMA LOSS + NO ALL LOSS"
