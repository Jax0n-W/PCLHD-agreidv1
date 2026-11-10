# -*- coding: utf-8 -*-
"""
PCLHD LAGPeR Test Script (SeCap Protocol)

Evaluates A2G, G2A, G2A+G modes on LAGPeR dataset using SeCap .txt protocol files.

LAGPeR dataset structure:
  query_aerial/              : aerial (IR) query images
  query_ground/              : ground (RGB) query images
  bounding_box_test_aerial/  : aerial (IR) gallery images
  bounding_box_test_ground/  : ground (RGB) gallery images
  bounding_box_gallery/      : additional gallery images
  bounding_box_train/        : training images

SeCap protocol files:
  exp3_A2G+.txt  : Aerial query -> Ground gallery
  exp4_G2A+.txt  : Ground query -> Aerial gallery
  exp5_G2A+G.txt : Ground query -> Aerial + Ground gallery

LAGPeR filename convention: {pid}_c{camid}.jpg  (e.g. "P0001_c1.jpg")
  - Aerial/IR cameras typically have camid in [0,1,2]  (modal=2)
  - Ground/RGB cameras typically have camid=3           (modal=1)
  - Modal is determined by the folder name (aerial vs ground), not camid

Usage:
    python examples/PCLHD_LAGPeR_g2ag.py \
        --data-dir /home/lab338/Jaxon/dataset/LAGPeR_test \
        --resume /home/lab338/Jaxon/PCLHD2/logs/lag_agva050_s2/1/model_best.pth.tar \
        --gpu 0
"""
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os
import re
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
from clustercontrast.models.agw_lag_stable import agw_lag_stable
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor
from clustercontrast.utils.serialization import load_checkpoint


def create_model(args):
    """Create the checkpoint-compatible model used by LAGPeR training."""
    if args.arch == 'agw':
        model = agw_lag_stable(
            num_features=args.features, norm=True,
            dropout=args.dropout, num_classes=0,
            pooling_type=args.pooling_type)
    else:
        model = models.create(
            args.arch, num_features=args.features, norm=True,
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
        modal: 1 = RGB (Ground), 2 = IR (Aerial)
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


def parse_lagper_pid_and_camid(fname):
    """
    Extract PID and camera ID from LAGPeR filename.

    SeCap uses pattern: `r'([-\\d]+)_c([-\\d]+)'` and matches formats like:
      - "P0001_c1.jpg"       -> pid=1, camid=1
      - "2332_c19_000009.jpg" -> pid=2332, camid=19
      - "-1_c1_000004.jpg"   -> pid=-1, camid=1 (distractor/junk, kept as noise)

    Returns:
        (pid, camid) as ints, or (None, None) if parse fails
    """
    # SeCap-style pattern: match any sequence of digits (including leading '-') after "P" or at start
    pattern = re.compile(r'([-\d]+)_c([-\d]+)')
    m = pattern.search(fname)
    if m is not None:
        pid = int(m.group(1))
        camid = int(m.group(2))
        return pid, camid
    return None, None


def map_camid(camid):
    """
    Map camera ID according to SeCap LAGPeR convention.

    SeCap convention:
      - camid in [3, 6, 9, 13, 16, 19, 22] -> 'Aerial' (IR)
      - otherwise -> 'Ground' (RGB)
      - If camid > 10: camid -= 1

    Returns:
        (mapped_camid, is_aerial)
    """
    if camid in [3, 6, 9, 13, 16, 19, 22]:
        is_aerial = True
    else:
        is_aerial = False
    if camid > 10:
        camid -= 1
    return camid, is_aerial


def is_aerial_from_folder(folder_name):
    """
    Determine if a folder contains aerial (IR) or ground (RGB) images.

    Folder naming:
      - 'aerial' in folder name -> aerial/IR (modal=2)
      - 'ground' in folder name -> ground/RGB (modal=1)

    Returns:
        True if aerial/IR, False if ground/RGB
    """
    folder_lower = folder_name.lower()
    if 'aerial' in folder_lower:
        return True
    elif 'ground' in folder_lower:
        return False
    else:
        # Default: if folder is named 'query' or 'gallery' without aerial/ground,
        # we need to be more careful. Check for 'ir' in folder name.
        if 'ir' in folder_lower or 'thermal' in folder_lower:
            return True
        return False


def parse_secap_protocol_entry(rel_path, data_dir):
    """
    Parse a single line from a SeCap protocol .txt file.

    The line is a relative path like:
      query_aerial/P0001_c1.jpg
      bounding_box_test_ground/P0007_c3.jpg

    SeCap LAGPeR behavior:
      - PID=-1 (distractor/junk) images are KEPT as gallery noise
      - Follows original SeCap logic: if pid == -1: do NOT skip

    Returns:
        (abs_path, pid, camid, is_aerial, is_junk) or None if parsing/timeout
    """
    rel_path = rel_path.strip().replace('\\', '/')
    abs_path = osp.join(data_dir, rel_path)

    if not osp.exists(abs_path):
        # File not found - try without the leading newline
        print("  ⚠ File not found: {}".format(abs_path))
        return None

    folder_name = osp.basename(osp.dirname(rel_path))
    fname = osp.basename(abs_path)

    pid, camid = parse_lagper_pid_and_camid(fname)
    if pid is None:
        print("  ⚠ Cannot parse PID/camid from filename: {}  [skipping]".format(fname))
        return None

    # Map camid according to SeCap convention
    mapped_camid, is_aerial = map_camid(camid)
    is_junk = (pid == -1)

    return (abs_path, pid, mapped_camid, is_aerial, is_junk)


def parse_secap_protocol_txt(txt_path, data_dir):
    """
    Read SeCap protocol .txt file, parse query and gallery entries.

    The file contains relative paths, one per line.
    Query entries are those with 'query' in the folder path.
    Gallery entries are those with 'gallery' or 'bounding_box_test' in the folder path.

    Returns:
        query_list: list of (abs_path, pid, camid, is_aerial)
        gallery_list: list of (abs_path, pid, camid, is_aerial)
    """
    if not osp.exists(txt_path):
        print("  ⚠ Protocol file not found: {}".format(txt_path))
        return [], []

    query_list = []
    gallery_list = []

    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            entry = parse_secap_protocol_entry(line, data_dir)
            if entry is None:
                continue

            # entry format: (abs_path, pid, mapped_camid, is_aerial, is_junk)
            #   - is_junk (pid==-1) images are KEPT as noise in gallery
            abs_path, pid, camid, is_aerial, is_junk = entry
            rel_path = line.replace('\\', '/')

            # Protocol file format: <top_folder>/<pid_subfolder>/<filename>.jpg
            #   e.g.  query_aerial/2332/2332_c19_000009.jpg
            #         bounding_box_test_ground/2332/2332_c19_000003.jpg
            #   The top-level folder determines query vs gallery.
            top_folder = rel_path.split('/')[0]

            # Determine if query or gallery from top-level folder name
            # SeCap convention: 'query_*' folders are query,
            #   'bounding_box_test_*', 'bounding_box_gallery', 'gallery_*' are gallery
            if top_folder.startswith('query_'):
                query_list.append((abs_path, pid, camid, is_aerial))
            elif (top_folder.startswith('bounding_box_test_')
                  or top_folder.startswith('bounding_box_gallery')
                  or top_folder.startswith('gallery_')):
                gallery_list.append((abs_path, pid, camid, is_aerial))
            else:
                # Fallback: check if the folder name contains 'query'
                if 'query' in top_folder.lower():
                    query_list.append((abs_path, pid, camid, is_aerial))
                else:
                    gallery_list.append((abs_path, pid, camid, is_aerial))

    return query_list, gallery_list


def eval_lagper(distmat, q_pids, g_pids, max_rank=20, q_camids=None, g_camids=None):
    """
    Standard ReID evaluation (same as test_LAG.py's eval_lagper).

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
    Evaluate a single mode (A2G, G2A, or G2A+G).

    Args:
        model: PCLHD agw model
        query_entries: list of (abs_path, pid, camid, is_aerial)
        gall_entries: list of (abs_path, pid, camid, is_aerial)
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
    # Determine modal from aerial/ground
    # is_aerial=True -> IR -> modal=2, is_aerial=False -> RGB -> modal=1
    query_modal = np.array([2 if e[3] else 1 for e in query_entries])

    # --- Parse gallery entries ---
    gall_paths = [e[0] for e in gall_entries]
    gall_pids = np.array([e[1] for e in gall_entries])
    gall_camids = np.array([e[2] for e in gall_entries])
    gall_modal = np.array([2 if e[3] else 1 for e in gall_entries])

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
    cmc, mAP, mINP = eval_lagper(-distmat, query_pids, gall_pids,
                                  q_camids=query_camids, g_camids=gall_camids)

    print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%} | Rank-20: {:.2%} | mAP: {:.2%} | mINP: {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

    return cmc, mAP, mINP


def main():
    parser = argparse.ArgumentParser(description="PCLHD LAGPeR Test (SeCap Protocol)")
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to LAGPeR_test dataset, '
                             'e.g. /home/lab338/Jaxon/dataset/LAGPeR_test')
    parser.add_argument('--resume', type=str, required=True,
                        help='Path to checkpoint.pth.tar, '
                             'e.g. /home/lab338/Jaxon/PCLHD-main/logs/lag_s1/1/checkpoint.pth.tar')
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
    # Protocol txt file paths (SeCap convention)
    parser.add_argument('--a2g-txt', type=str, default=None,
                        help='Path to exp3_A2G+.txt (default: <data-dir>/exp3_A2G+.txt)')
    parser.add_argument('--g2a-txt', type=str, default=None,
                        help='Path to exp4_G2A+.txt (default: <data-dir>/exp4_G2A+.txt)')
    parser.add_argument('--g2ag-txt', type=str, default=None,
                        help='Path to exp5_G2A+G.txt (default: <data-dir>/exp5_G2A+G.txt)')
    args = parser.parse_args()

    cudnn.benchmark = True

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print("==> Using GPU: {}".format(args.gpu))

    # Setup logger
    script_dir = osp.dirname(osp.abspath(__file__))
    log_path = osp.join(script_dir, 'PCLHD_LAGPeR_g2ag_log.txt')
    sys.stdout = open(log_path, 'w')
    print("Log saved to: {}".format(log_path))

    # Protocol file paths (SeCap convention for LAGPeR)
    a2g_txt = args.a2g_txt or osp.join(args.data_dir, 'exp3_A2G+.txt')
    g2a_txt = args.g2a_txt or osp.join(args.data_dir, 'exp4_G2A+.txt')
    g2ag_txt = args.g2ag_txt or osp.join(args.data_dir, 'exp5_G2A+G.txt')

    # ========================================================
    # 1. Parse SeCap protocol files
    # ========================================================
    print("==> Parse SeCap protocol txt files from: {}".format(args.data_dir))

    for fpath, name in [(a2g_txt, 'exp3_A2G+'), (g2a_txt, 'exp4_G2A+'), (g2ag_txt, 'exp5_G2A+G')]:
        if not osp.exists(fpath):
            print("  ⚠ Protocol file not found: {} ({})".format(fpath, name))
        else:
            print("  ✓ Found: {} ({})".format(fpath, name))

    a2g_q, a2g_g = parse_secap_protocol_txt(a2g_txt, args.data_dir)
    g2a_q, g2a_g = parse_secap_protocol_txt(g2a_txt, args.data_dir)
    g2ag_q, g2ag_g = parse_secap_protocol_txt(g2ag_txt, args.data_dir)

    print("  exp3_A2G+:   query={}, gallery={}".format(len(a2g_q), len(a2g_g)))
    print("  exp4_G2A+:   query={}, gallery={}".format(len(g2a_q), len(g2a_g)))
    print("  exp5_G2A+G:  query={}, gallery={}".format(len(g2ag_q), len(g2ag_g)))

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
    # 4. Evaluate 3 modes (SeCap protocol for LAGPeR)
    # ========================================================
    img_size = (args.width, args.height)

    results = {}

    # --- Mode A2G: Aerial query -> Ground gallery (exp3_A2G+.txt) ---
    print("\n" + "#" * 60)
    print("# Mode 1: Aerial -> Ground (A2G)")
    print("#   Protocol file: exp3_A2G+.txt")
    print("#   Query  = aerial (IR, modal=2)")
    print("#   Gallery = ground (RGB, modal=1)")
    print("#" * 60)
    cmc_a2g, mAP_a2g, mINP_a2g = evaluate_mode(
        model, a2g_q, a2g_g, transform_test, args, "A2G", img_size)
    results['A2G'] = (cmc_a2g, mAP_a2g, mINP_a2g)

    # --- Mode G2A: Ground query -> Aerial gallery (exp4_G2A+.txt) ---
    print("\n" + "#" * 60)
    print("# Mode 2: Ground -> Aerial (G2A)")
    print("#   Protocol file: exp4_G2A+.txt")
    print("#   Query  = ground (RGB, modal=1)")
    print("#   Gallery = aerial (IR, modal=2)")
    print("#" * 60)
    cmc_g2a, mAP_g2a, mINP_g2a = evaluate_mode(
        model, g2a_q, g2a_g, transform_test, args, "G2A", img_size)
    results['G2A'] = (cmc_g2a, mAP_g2a, mINP_g2a)

    # --- Mode G2A+G: Ground query -> Aerial + Ground gallery (exp5_G2A+G.txt) ---
    print("\n" + "#" * 60)
    print("# Mode 3: Ground -> Aerial + Ground (G->A+G)")
    print("#   Protocol file: exp5_G2A+G.txt")
    print("#   Query  = ground (RGB, modal=1)")
    print("#   Gallery = aerial (IR, modal=2) + ground (RGB, modal=1)")
    print("#" * 60)
    cmc_g2ag, mAP_g2ag, mINP_g2ag = evaluate_mode(
        model, g2ag_q, g2ag_g, transform_test, args, "G->A+G", img_size)
    results['G2A+G'] = (cmc_g2ag, mAP_g2ag, mINP_g2ag)

    # ========================================================
    # 5. Summary
    # ========================================================
    print("\n" + "=" * 70)
    print("SUMMARY - LAGPeR SeCap Protocol (PCLHD)")
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