"""Confidence-gated cross-view prototype matching for AG-ReID Stage 2.

The released baseline forces every cluster into the bipartite mapping.  This
module keeps only reciprocal nearest-neighbour prototype pairs that also pass
an absolute cosine-similarity threshold and a bidirectional ambiguity margin.
It has no dataset, clustering, or memory-bank side effects.
"""
from __future__ import absolute_import, print_function

import torch
import torch.nn.functional as F


def _second_best_margin(similarity, dim):
    """Return best minus second-best values along ``dim``.

    A singleton opposite domain is unambiguous, so its margin is treated as
    positive infinity instead of rejecting the only possible pair.
    """
    size = similarity.size(dim)
    best = similarity.max(dim=dim).values
    if size == 1:
        return torch.full_like(best, float("inf"))
    top2 = similarity.topk(2, dim=dim).values
    return top2.select(dim, 0) - top2.select(dim, 1)


@torch.no_grad()
def build_confident_cluster_matches(rgb_prototypes, ir_prototypes,
                                    min_similarity=0.30,
                                    min_margin=0.02):
    """Build partial RGB<->IR maps from reliable prototype pairs.

    Returns ``(i2r, r2i, stats)``.  Keys and values are Python integers so the
    maps can be consumed directly by the existing pseudo-label training loop.
    Unlike the baseline Hungarian assignment, both maps are intentionally
    partial: samples whose cluster is absent must receive no cross-view loss.
    """
    if rgb_prototypes.ndim != 2 or ir_prototypes.ndim != 2:
        raise ValueError("prototype tensors must be two-dimensional")
    if rgb_prototypes.size(1) != ir_prototypes.size(1):
        raise ValueError("RGB and IR prototype dimensions must match")
    if not -1.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be in [-1, 1]")
    if min_margin < 0.0:
        raise ValueError("min_margin must be non-negative")

    num_rgb = int(rgb_prototypes.size(0))
    num_ir = int(ir_prototypes.size(0))
    empty_stats = {
        "num_rgb": num_rgb,
        "num_ir": num_ir,
        "mutual_pairs": 0,
        "accepted_pairs": 0,
        "rgb_coverage": 0.0,
        "ir_coverage": 0.0,
        "mean_similarity": 0.0,
    }
    if num_rgb == 0 or num_ir == 0:
        return {}, {}, empty_stats

    rgb = F.normalize(rgb_prototypes.detach(), dim=1)
    ir = F.normalize(ir_prototypes.detach(), dim=1)
    similarity = torch.mm(rgb, ir.t())
    rgb_best_ir = similarity.argmax(dim=1)
    ir_best_rgb = similarity.argmax(dim=0)
    rgb_margin = _second_best_margin(similarity, dim=1)
    ir_margin = _second_best_margin(similarity, dim=0)

    accepted = []
    mutual_pairs = 0
    for rgb_id in range(num_rgb):
        ir_id = int(rgb_best_ir[rgb_id].item())
        if int(ir_best_rgb[ir_id].item()) != rgb_id:
            continue
        mutual_pairs += 1
        score = float(similarity[rgb_id, ir_id].item())
        if score < min_similarity:
            continue
        if float(rgb_margin[rgb_id].item()) < min_margin:
            continue
        if float(ir_margin[ir_id].item()) < min_margin:
            continue
        accepted.append((rgb_id, ir_id, score))

    r2i = {rgb_id: ir_id for rgb_id, ir_id, _ in accepted}
    i2r = {ir_id: rgb_id for rgb_id, ir_id, _ in accepted}
    stats = dict(empty_stats)
    stats.update({
        "mutual_pairs": mutual_pairs,
        "accepted_pairs": len(accepted),
        "rgb_coverage": len(accepted) / float(num_rgb),
        "ir_coverage": len(accepted) / float(num_ir),
        "mean_similarity": (
            sum(score for _, _, score in accepted) / len(accepted)
            if accepted else 0.0),
    })
    return i2r, r2i, stats


def alignment_weight_at_epoch(epoch, target_weight, warmup_epochs,
                              ramp_epochs):
    """Linear post-warmup schedule for the reliable cross-view loss."""
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if target_weight < 0.0:
        raise ValueError("target_weight must be non-negative")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")
    if ramp_epochs < 1:
        raise ValueError("ramp_epochs must be at least 1")
    if epoch < warmup_epochs:
        return 0.0
    progress = min(1.0, (epoch - warmup_epochs + 1) / float(ramp_epochs))
    return target_weight * progress

