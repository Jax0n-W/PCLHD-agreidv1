# -*- coding: utf-8 -*-
"""
PCLHD AG-ReID Test Script (SeCap Protocol)

Evaluates A2G, G2A, G2A+G modes on AG-ReID dataset.

AG-ReID naming convention at AGreid_test:
  - query_all_c0/      : aerial (IR, camera 0) query images
  - query_all_c3/      : ground (RGB, camera 3) query images
  - bounding_box_test_all_c0/ : aerial (IR, camera 0) gallery images
  - bounding_box_test_all_c3/ : ground (RGB, camera 3) gallery images

SeCap testing modes:
  - A2G:  Aerial query (C00) -> Ground gallery (C03)
  - G2A:  Ground query (C03) -> Aerial gallery (C00)
  - G2A+G: Ground query (C03) -> Aerial gallery (C00) + Ground gallery (C03)

Usage:
    python examples/PCLHD_AGReID_g2ag.py \
        --data-dir /home/lab338/Jaxon/dataset/AGreid_test \
        --resume /home/lab338/Jaxon/PCLHD2/logs_agva050/agreid_s2/1/model_best.pth.tar \
        --gpu 0
"""
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os
import sys
import time
import numpy as np
from PIL import Image

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.utils.data as data
from torch.autograd import Variable

from clustercontrast import models
from agreid_identity import (
    AGREID_TEST_IDENTITIES, parse_agreid_name)
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor
from clustercontrast.utils.serialization import load_checkpoint


def create_model(args):
    """Create PCLHD model (agw) — identical to train_agreid.py."""
    model = models.create(args.arch, num_features=args.features, norm=True,
                          dropout=args.dropout, num_classes=0,
                          pooling_type=args.pooling_type)
    model.cuda()
    model = nn.DataParallel(model)
    return model


class TestData(data.Dataset):
    """Dataset that loads images on demand from path list (lazy loading)."""
    def __init__(self, test_img_file, test_label, transform=None, img_size=(144, 288)):
        self.test_img_file = test_img_file
        self.test_label = test_label
        self.transform = transform
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.test_img_file[index]
        target = self.test_label[index]
        img = Image.open(img_path)
        img = img.resize((self.img_size[0], self.img_size[1]), Image.LANCZOS)
        img_array = np.array(img)
        img_tensor = self.transform(img_array)
        return img_tensor, target

    def __len__(self):
        return len(self.test_img_file)


def fliplr(img):
    """Horizontal flip."""
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feat_modal(model, loader, n, modal=1):
    """
    Extract features with specified modal.

    Args:
        model: PCLHD agw model
        loader: DataLoader
        n: number of samples
        modal: 1 = RGB (Ground/C03), 2 = IR (Aerial/C00)
    """
    pool_dim = 2048
    net = model
    net.eval()
    modal_name = 'RGB(Ground)' if modal == 1 else 'IR(Aerial)'
    print('Extracting features (modal={}: {})...'.format(modal, modal_name))
    start = time.time()
    ptr = 0
    feat_fc = np.zeros((n, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc_out = net(input, input, modal)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1 = net(flip_input, flip_input, modal)
            feature_fc = (feat_fc_out.detach() + feat_fc_1.detach()) / 2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            feat_fc[ptr:ptr + batch_num, :] = feature_fc.cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time: {:.3f}'.format(time.time() - start))
    return feat_fc


def parse_agreid_pid(fname):
    """
    Extract PID from AG-ReID filename.

    AG-ReID format: P{pid}T{tracklet}A...C{cam}F{frame}.jpg
    The official identity is P, T and A concatenated; P alone is not unique.

    Returns:
        pid: person ID (int), or None if parse fails
    """
    try:
        return parse_agreid_name(fname).pid
    except ValueError:
        return None


def validate_agreid_test_identity_count(*entry_groups):
    identities = {
        int(pid) for group in entry_groups for _, pid, _ in group}
    if len(identities) != AGREID_TEST_IDENTITIES:
        raise RuntimeError(
            "AG-ReID test identity count mismatch: expected {}, found {}. "
            "The official identity must concatenate P/T/A.".format(
                AGREID_TEST_IDENTITIES, len(identities)))


def camid_from_folder(folder_name):
    """
    Determine camera ID from folder name suffix.

    Folder naming:  query_all_c0  -> camera 0 (Aerial/IR)
                    query_all_c3  -> camera 3 (Ground/RGB)
    """
    if folder_name.endswith('_c0'):
        return 0
    elif folder_name.endswith('_c3'):
        return 3
    else:
        return -1


def load_images_from_folder(folder_path, data_dir):
    """
    Load all images from a folder, parse PID from filename.
    Camera ID is determined from the folder name, not the filename.

    Returns:
        entries: list of (abs_path, pid, camid)
    """
    full_path = osp.join(data_dir, folder_path)
    entries = []
    if not osp.isdir(full_path):
        print("  ⚠ Directory not found: {}".format(full_path))
        return entries

    camid = camid_from_folder(folder_path)
    if camid == -1:
        print("  ⚠ Unknown camera from folder: {}".format(folder_path))
        return entries

    for fname in sorted(os.listdir(full_path)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        pid = parse_agreid_pid(fname)
        if pid is None:
            print("  ⚠ Cannot parse PID from filename: {}".format(fname))
            continue
        abs_path = osp.join(full_path, fname)
        entries.append((abs_path, pid, camid))
    return entries


def eval_agreid(distmat, q_pids, g_pids, max_rank=20, q_camids=None, g_camids=None):
    """
    Standard ReID evaluation (same as test_agreid.py's eval_agreid).

    Args:
        distmat: distance matrix (negative cosine similarity, so smaller = more similar)
        q_pids: query person IDs
        g_pids: gallery person IDs
        max_rank: maximum rank for CMC
        q_camids: query camera IDs (used to filter same-camera same-ID)
        g_camids: gallery camera IDs
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.

    if q_camids is None:
        q_camids = np.ones(num_q).astype(np.int32)
    if g_camids is None:
        g_camids = 2 * np.ones(num_g).astype(np.int32)

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            continue
        cmc = raw_cmc.cumsum()
        pos_idx = np.where(raw_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)
        all_INP.append(inp)
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return all_cmc, mAP, mINP


def evaluate_mode(model, query_entries, gall_entries, transform_test, args, mode_name, img_size=(144, 288)):
    """
    Evaluate a single mode.

    Args:
        model: PCLHD agw model
        query_entries: list of (abs_path, pid, camid)
        gall_entries: list of (abs_path, pid, camid)
        transform_test: image transform
        args: command line args
        mode_name: e.g. "A2G", "G2A", "G->A+G"
        img_size: (width, height)
    """
    print("\n" + "=" * 60)
    print("Mode: {} | query={} gallery={}".format(mode_name, len(query_entries), len(gall_entries)))
    print("=" * 60)

    if len(query_entries) == 0 or len(gall_entries) == 0:
        print("  ⚠ Skipping: empty query or gallery!")
        return None, None, None

    # --- Parse query entries ---
    query_paths = [e[0] for e in query_entries]
    query_pids = np.array([e[1] for e in query_entries])
    query_camids = np.array([e[2] for e in query_entries])
    # Determine modal: camid==0 -> Aerial(IR) -> modal=2, camid==3 -> Ground(RGB) -> modal=1
    query_modal = np.array([2 if c == 0 else 1 for c in query_camids])

    # --- Parse gallery entries ---
    gall_paths = [e[0] for e in gall_entries]
    gall_pids = np.array([e[1] for e in gall_entries])
    gall_camids = np.array([e[2] for e in gall_entries])
    gall_modal = np.array([2 if c == 0 else 1 for c in gall_camids])

    # --- Extract query features ---
    q_rgb_indices = np.where(query_modal == 1)[0]
    q_ir_indices = np.where(query_modal == 2)[0]

    query_feat = np.zeros((len(query_paths), 2048), dtype=np.float32)

    if len(q_rgb_indices) > 0:
        q_rgb_paths = [query_paths[i] for i in q_rgb_indices]
        q_rgb_labels = np.zeros(len(q_rgb_paths))  # dummy labels
        rgb_set = TestData(q_rgb_paths, q_rgb_labels, transform=transform_test, img_size=img_size)
        rgb_loader = data.DataLoader(rgb_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        rgb_feat = extract_feat_modal(model, rgb_loader, len(q_rgb_paths), modal=1)
        for idx_in_batch, idx_in_query in enumerate(q_rgb_indices):
            query_feat[idx_in_query] = rgb_feat[idx_in_batch]

    if len(q_ir_indices) > 0:
        q_ir_paths = [query_paths[i] for i in q_ir_indices]
        q_ir_labels = np.zeros(len(q_ir_paths))
        ir_set = TestData(q_ir_paths, q_ir_labels, transform=transform_test, img_size=img_size)
        ir_loader = data.DataLoader(ir_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        ir_feat = extract_feat_modal(model, ir_loader, len(q_ir_paths), modal=2)
        for idx_in_batch, idx_in_query in enumerate(q_ir_indices):
            query_feat[idx_in_query] = ir_feat[idx_in_batch]

    # --- Extract gallery features ---
    g_rgb_indices = np.where(gall_modal == 1)[0]
    g_ir_indices = np.where(gall_modal == 2)[0]

    gall_feat = np.zeros((len(gall_paths), 2048), dtype=np.float32)

    if len(g_rgb_indices) > 0:
        g_rgb_paths = [gall_paths[i] for i in g_rgb_indices]
        g_rgb_labels = np.zeros(len(g_rgb_paths))
        rgb_set = TestData(g_rgb_paths, g_rgb_labels, transform=transform_test, img_size=img_size)
        rgb_loader = data.DataLoader(rgb_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        rgb_feat = extract_feat_modal(model, rgb_loader, len(g_rgb_paths), modal=1)
        for idx_in_batch, idx_in_gall in enumerate(g_rgb_indices):
            gall_feat[idx_in_gall] = rgb_feat[idx_in_batch]

    if len(g_ir_indices) > 0:
        g_ir_paths = [gall_paths[i] for i in g_ir_indices]
        g_ir_labels = np.zeros(len(g_ir_paths))
        ir_set = TestData(g_ir_paths, g_ir_labels, transform=transform_test, img_size=img_size)
        ir_loader = data.DataLoader(ir_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        ir_feat = extract_feat_modal(model, ir_loader, len(g_ir_paths), modal=2)
        for idx_in_batch, idx_in_gall in enumerate(g_ir_indices):
            gall_feat[idx_in_gall] = ir_feat[idx_in_batch]

    # --- Compute cosine similarity ---
    distmat = np.matmul(query_feat, np.transpose(gall_feat))

    # --- Evaluate (pass -distmat because eval expects smaller = more similar) ---
    cmc, mAP, mINP = eval_agreid(-distmat, query_pids, gall_pids,
                                  q_camids=query_camids, g_camids=gall_camids)

    print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%} | Rank-20: {:.2%} | mAP: {:.2%} | mINP: {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

    return cmc, mAP, mINP


def main():
    parser = argparse.ArgumentParser(description="PCLHD AG-ReID Test (SeCap Protocol)")
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to AGreid_test dataset, '
                             'e.g. /home/lab338/Jaxon/dataset/AGreid_test')
    parser.add_argument(
        '--resume', type=str, required=True,
        help='Path to checkpoint.pth.tar (or a legacy best checkpoint)')
    parser.add_argument('--arch', type=str, default='agw')
    parser.add_argument('--height', type=int, default=288)
    parser.add_argument('--width', type=int, default=144)
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU id to use, e.g. 0 or 1')
    parser.add_argument('--features', type=int, default=2048,
                        help='number of feature dimensions')
    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout ratio')
    args = parser.parse_args()

    cudnn.benchmark = True

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print("==> Using GPU: {}".format(args.gpu))

    # Setup logger
    script_dir = osp.dirname(osp.abspath(__file__))
    log_path = osp.join(script_dir, 'PCLHD_AGReID_g2ag_log.txt')
    sys.stdout = open(log_path, 'w')
    print("Log saved to: {}".format(log_path))

    # ========================================================
    # 1. Load images from AGreid_test subdirectories
    # ========================================================
    print("==> Loading AG-ReID dataset from: {}".format(args.data_dir))

    # Aerial (IR, camera=C00) images
    aerial_query = load_images_from_folder('query_all_c0', args.data_dir)
    aerial_gallery = load_images_from_folder('bounding_box_test_all_c0', args.data_dir)

    # Ground (RGB, camera=C03) images
    ground_query = load_images_from_folder('query_all_c3', args.data_dir)
    ground_gallery = load_images_from_folder('bounding_box_test_all_c3', args.data_dir)
    validate_agreid_test_identity_count(
        aerial_query, aerial_gallery, ground_query, ground_gallery)

    print("  query_all_c0 (Aerial/IR):       {} images".format(len(aerial_query)))
    print("  query_all_c3 (Ground/RGB):      {} images".format(len(ground_query)))
    print("  bounding_box_test_all_c0 (Aerial/IR):   {} images".format(len(aerial_gallery)))
    print("  bounding_box_test_all_c3 (Ground/RGB):  {} images".format(len(ground_gallery)))

    # ========================================================
    # 2. Build model and load checkpoint
    # ========================================================
    print("\n==> Build model: {}".format(args.arch))
    model = create_model(args)
    print("==> Load checkpoint from '{}'".format(args.resume))
    checkpoint = load_checkpoint(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # ========================================================
    # 3. Transform
    # ========================================================
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    transform_test = T.Compose([
        T.ToPILImage(),
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        normalize,
    ])

    # ========================================================
    # 4. Evaluate 3 modes (SeCap protocol)
    # ========================================================
    img_size = (args.width, args.height)

    results = {}

    # --- Mode A2G: Aerial query -> Ground gallery ---
    print("\n" + "#" * 60)
    print("# Mode 1: Aerial -> Ground (A2G)")
    print("#   Query  = query_all_c0 (Aerial/IR, camera 0)")
    print("#   Gallery = bounding_box_test_all_c3 (Ground/RGB, camera 3)")
    print("#" * 60)
    cmc_a2g, mAP_a2g, mINP_a2g = evaluate_mode(
        model, aerial_query, ground_gallery, transform_test, args, "A2G", img_size)
    results['A2G'] = (cmc_a2g, mAP_a2g, mINP_a2g)

    # --- Mode G2A: Ground query -> Aerial gallery ---
    print("\n" + "#" * 60)
    print("# Mode 2: Ground -> Aerial (G2A)")
    print("#   Query  = query_all_c3 (Ground/RGB, camera 3)")
    print("#   Gallery = bounding_box_test_all_c0 (Aerial/IR, camera 0)")
    print("#" * 60)
    cmc_g2a, mAP_g2a, mINP_g2a = evaluate_mode(
        model, ground_query, aerial_gallery, transform_test, args, "G2A", img_size)
    results['G2A'] = (cmc_g2a, mAP_g2a, mINP_g2a)

    # --- Mode G2A+G: Ground query -> Aerial + Ground gallery ---
    print("\n" + "#" * 60)
    print("# Mode 3: Ground -> Aerial + Ground (G->A+G)")
    print("#   Query  = query_all_c3 (Ground/RGB, camera 3)")
    print("#   Gallery = bounding_box_test_all_c0 (Aerial/IR) + bounding_box_test_all_c3 (Ground/RGB)")
    print("#" * 60)
    # Combine aerial and ground galleries
    combined_gallery = aerial_gallery + ground_gallery
    cmc_g2ag, mAP_g2ag, mINP_g2ag = evaluate_mode(
        model, ground_query, combined_gallery, transform_test, args, "G->A+G", img_size)
    results['G2A+G'] = (cmc_g2ag, mAP_g2ag, mINP_g2ag)

    # ========================================================
    # 5. Summary
    # ========================================================
    print("\n" + "=" * 70)
    print("SUMMARY - AG-ReID SeCap Protocol (PCLHD)")
    print("=" * 70)
    print("{:<10} {:<8} {:<8} {:<9} {:<9} {:<8} {:<8}".format(
        "Mode", "R-1", "R-5", "R-10", "R-20", "mAP", "mINP"))
    print("-" * 70)
    for mode_name in ['A2G', 'G2A', 'G2A+G']:
        cmc, mAP, mINP = results[mode_name]
        if cmc is not None:
            print("{:<10} {:<8.2%} {:<8.2%} {:<9.2%} {:<9.2%} {:<8.2%} {:<8.2%}".format(
                mode_name, cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        else:
            print("{:<10} {:<8} {:<8} {:<9} {:<9} {:<8} {:<8}".format(
                mode_name, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"))
    print("-" * 70)
    print("Log saved to: {}".format(log_path))


if __name__ == '__main__':
    main()
