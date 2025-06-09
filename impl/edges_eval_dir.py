import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from shutil import rmtree
from scipy.io import loadmat
from scipy.interpolate import interp1d
from joblib import Parallel, delayed

from .bwmorph_thin import bwmorph_thin
from .correspond_pixels import correspond_pixels

eps = 2e-6


def edges_eval_img(im, gt, out="", thrs=99, max_dist=0.0075, thin=True, need_v=False, workers=1):
    eps = 2e-16

    if isinstance(thrs, list):
        k = len(thrs)
        thrs = np.array(thrs)
    elif isinstance(thrs, int):
        k = thrs
        thrs = np.linspace(1 / (k + 1), 1 - 1 / (k + 1), k)
    else:
        raise NotImplementedError

    if isinstance(im, str):
        edge = cv2.imread(im, cv2.IMREAD_UNCHANGED) / 255.
    else:
        edge = im
    assert edge.ndim == 2

    # Cargar GT
    if isinstance(gt, str) and gt.endswith(".mat"):
        try:
            gt = [g.item()[1] for g in loadmat(gt)["groundTruth"][0]]
        except:
            gt = [g.item()[0] for g in loadmat(gt)["groundTruth"][0]]
    elif isinstance(gt, str) and gt.endswith(".png"):
        gt_img = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
        gt_bin = (gt_img > 127).astype(np.uint8)  # blanco = borde
        gt = [gt_bin]
    else:
        raise ValueError(f"Formato GT no soportado: {gt}")

    def eval_threshold(k_):
        e1 = edge >= max(eps, thrs[k_])
        if thin:
            e1 = bwmorph_thin(e1)

        match_e = np.zeros_like(edge, dtype=bool)
        match_g = np.zeros_like(edge, dtype=np.int32)
        all_g = np.zeros_like(edge, dtype=np.int32)
        v_local = np.zeros((*edge.shape, 3), dtype=np.float32) if need_v else None

        for g in gt:
            g = g.astype(np.int32)  # 💡 Fix importante para evitar el error de casting
            match_e1, match_g1, _, _ = correspond_pixels(e1, g, max_dist)
            match_e = np.logical_or(match_e, match_e1 > 0)
            match_g = match_g + (match_g1 > 0)
            all_g += g

            if need_v:
                v_local[:, :, 0] += match_g1 > 0
                v_local[:, :, 1] += match_e1 > 0
                v_local[:, :, 2] += np.logical_and(match_e1 > 0, match_g1 > 0)

        cnts = [
            np.sum(match_g),
            np.sum(all_g),
            np.count_nonzero(match_e),
            np.count_nonzero(e1)
        ]
        return k_, cnts, v_local

    results = Parallel(n_jobs=workers)(
        delayed(eval_threshold)(k_) for k_ in range(k)
    )

    results.sort(key=lambda x: x[0])
    cnt_sum_r_p = np.array([r[1] for r in results], dtype=np.int32)

    v = None
    if need_v:
        v = np.stack([r[2] for r in results], axis=-1)

    info = np.concatenate([thrs[:, None], cnt_sum_r_p], axis=1)
    if out:
        np.savetxt(out, info, fmt="%10g")
    return info, v



def compute_rpf(cnt_sum_r_p):
    r = cnt_sum_r_p[:, 0] / np.maximum(eps, cnt_sum_r_p[:, 1])
    p = cnt_sum_r_p[:, 2] / np.maximum(eps, cnt_sum_r_p[:, 3])
    f = 2 * p * r / np.maximum(eps, p + r)
    return r, p, f


def find_best_rpf(t, r, p):
    if len(t) == 1:
        bst_t, bst_r, bst_p = t, r, p
        bst_f = 2 * p * r / np.maximum(eps, p + r)
        return bst_r, bst_p, bst_f, bst_t
    a = np.linspace(0, 1, 100)[None, :]
    b = 1 - a
    t, r, p = t[:, None], r[:, None], p[:, None]
    rj = r[1:] @ a + r[:-1] @ b  # (len(T), len(A))
    pj = p[1:] @ a + p[:-1] @ b  # (len(T), len(A))
    tj = t[1:] @ a + t[:-1] @ b  # (len(T), len(A))
    fj = 2 * pj * rj / np.maximum(eps, pj + rj)
    k = np.argmax(fj).item()
    row, col = divmod(k, 100)
    bst_r, bst_p, bst_f, bst_t = rj[row, col], pj[row, col], fj[row, col], tj[row, col]
    return bst_r, bst_p, bst_f, bst_t


def edges_eval_dir(res_dir, gt_dir, cleanup=0, thrs=99, max_dist=0.0075, thin=True, workers=1):
    if thrs != 99:
        eval_dir = f"{res_dir}-eval-{thrs}"
    else:
        eval_dir = f"{res_dir}-eval"
    os.makedirs(eval_dir, exist_ok=True)
    filename = os.path.join(eval_dir, "eval_bdry.txt")
    if os.path.isfile(filename):
        return

    assert os.path.isdir(res_dir) and os.path.isdir(gt_dir)
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.mat")) + glob.glob(os.path.join(gt_dir, "*.png")))
    ids = [os.path.splitext(os.path.basename(f))[0] for f in gt_files]

    for i in tqdm(ids):
        res = os.path.join(eval_dir, f"{i}_ev.txt")
        if os.path.isfile(res):
            continue
        im = os.path.join(res_dir, f"{i}.png")
        gt_mat = os.path.join(gt_dir, f"{i}.mat")
        gt_png = os.path.join(gt_dir, f"{i}.png")
        gt = gt_mat if os.path.exists(gt_mat) else gt_png
        edges_eval_img(im, gt, out=res, thrs=thrs, max_dist=max_dist, thin=thin, workers=workers)

    cnt_sum_r_p = 0
    ois_cnt_sum_r_p = 0
    scores = np.zeros((len(ids), 5), dtype=np.float32)

    t = np.linspace(1 / (thrs + 1), 1 - 1 / (thrs + 1), thrs)

    for i, name in enumerate(ids):
        res = os.path.join(eval_dir, f"{name}_ev.txt")
        res = np.loadtxt(res, dtype=np.float32)
        t_vals, res = res[:, 0], res[:, 1:]
        cnt_sum_r_p += res
        r, p, f = compute_rpf(res)
        k = f.argmax()
        ois_r1, ois_p1, ois_f1, ois_t1 = find_best_rpf(t_vals, r, p)
        scores[i, :] = [i + 1, ois_t1, ois_r1, ois_p1, ois_f1]
        ois_cnt_sum_r_p += res[k, :]

    r, p, f = compute_rpf(cnt_sum_r_p)
    ods_r, ods_p, ods_f, ods_t = find_best_rpf(t, r, p)
    ois_r, ois_p, ois_f = compute_rpf(ois_cnt_sum_r_p[None, :])

    k = np.unique(r, return_index=True)[1][::-1]
    r, p, t, f, ap = r[k], p[k], t[k], f[k], 0
    if len(r) > 1:
        ap = interp1d(r, p, bounds_error=False, fill_value=0)(np.linspace(0, 1, 101))
        ap = np.sum(ap) / 100.0
    _, o = np.unique(p, return_index=True)
    r50 = interp1d(p[o], r[o], bounds_error=False, fill_value=np.nan)(np.maximum(p[o[0]], 0.5))

    bdry = np.array([[ods_t, ods_r, ods_p, ods_f, ois_r.item(), ois_p.item(), ois_f.item(), ap]])
    bdry_thr = np.stack([t, r, p, f], axis=0).T
    np.savetxt(os.path.join(eval_dir, "eval_bdry_img.txt"), scores, fmt="%.6f")
    np.savetxt(os.path.join(eval_dir, "eval_bdry_thr.txt"), bdry_thr, fmt="%.6f")
    np.savetxt(os.path.join(eval_dir, "eval_bdry.txt"), bdry, fmt="%.6f")
    print(f"ODS: {ods_f:.4f}    OIS: {ois_f.item():.4f}")

    if cleanup:
        for f in os.listdir(eval_dir):
            if f.endswith("_ev.txt"):
                os.remove(os.path.join(eval_dir, f))
        rmtree(res_dir)
