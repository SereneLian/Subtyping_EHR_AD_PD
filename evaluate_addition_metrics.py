"""
Cleaned clustering/evaluation script.

- Sensitive paths and model loading removed.
- Assumes embeddings (X_train, X_test, CPRD_X_test, UKB_X_test) are already available,
  or can be loaded from files specified with CLI args.
- Saves results to a local results directory by default (no hard-coded absolute paths).
"""

import os
import sys
import time
import random
import argparse
from math import sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, adjusted_rand_score,
    calinski_harabasz_score, confusion_matrix
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import KFold

# ----------------------------
# Small local helpers (avoids external utils import)
# ----------------------------
def create_folder(path):
    os.makedirs(path, exist_ok=True)

def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _maybe_sample(X, max_n, rs):
    X = _to_np(X)
    if len(X) <= max_n:
        return X
    idx = rs.choice(len(X), size=max_n, replace=False)
    return X[idx]

# ----------------------------
# Metrics / utilities
# ----------------------------
def silhouette_safe(X, labels, max_n=10000, random_state=0):
    rs = np.random.RandomState(random_state)
    X_np = _to_np(X)
    n = len(X_np)
    if n > max_n:
        idx = rs.choice(n, size=max_n, replace=False)
        return silhouette_score(X_np[idx], labels[idx], metric='euclidean')
    return silhouette_score(X_np, labels, metric='euclidean')

def gap_statistic(X, Ks, B=10, random_state=0, max_n=10000):
    """Tibshirani et al. JRSS-B 2001."""
    rs = np.random.RandomState(random_state)
    Xs = _maybe_sample(X, max_n, rs)
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(Xs)
    mins = Xs.min(axis=0)
    maxs = Xs.max(axis=0)

    gaps, sks, wk_log, wkref_log_mean = [], [], [], []
    for K in Ks:
        km = KMeans(n_clusters=K, n_init=10, max_iter=300, random_state=random_state).fit(Xs)
        wk = km.inertia_
        wk_log.append(np.log(wk))

        ref_logs = []
        for b in range(B):
            X_ref = rs.uniform(low=mins, high=maxs, size=Xs.shape)
            km_ref = KMeans(n_clusters=K, n_init=10, max_iter=300, random_state=random_state+b+1).fit(X_ref)
            ref_logs.append(np.log(km_ref.inertia_))
        ref_logs = np.array(ref_logs)
        gap = ref_logs.mean() - np.log(wk)
        s_k = ref_logs.std(ddof=1) * sqrt(1 + 1.0/B)
        gaps.append(gap); sks.append(s_k); wkref_log_mean.append(ref_logs.mean())
    return np.array(gaps), np.array(sks), np.array(wk_log), np.array(wkref_log_mean)

def comb2(n):
    return n*(n-1)//2

def jaccard_partition(labels_a, labels_b):
    """Pair-counting Jaccard for two partitions on the SAME items."""
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    tbl = pd.crosstab(labels_a, labels_b).to_numpy()
    m11 = np.sum(comb2(tbl))
    a = tbl.sum(axis=1)
    b = tbl.sum(axis=0)
    m10 = np.sum(comb2(a)) - m11
    m01 = np.sum(comb2(b)) - m11
    denom = (m11 + m10 + m01)
    if denom == 0:
        return 1.0
    return m11 / denom

def consensus_pac_monti(X, K, reps=30, subsample=0.8, sample_size=2000, random_state=0):
    """Monti consensus; returns PAC (lower better)."""
    rs = np.random.RandomState(random_state)
    Xs = _maybe_sample(X, sample_size, rs)
    n = len(Xs)
    C = np.zeros((n, n), dtype=np.float32)
    I = np.zeros((n, n), dtype=np.float32)

    for r in range(reps):
        m = max(2, int(subsample * n))
        idx = rs.choice(n, size=m, replace=False)
        km = KMeans(n_clusters=K, n_init=10, max_iter=300, random_state=random_state+r).fit(Xs[idx])
        labs = km.labels_
        same = (labs.reshape(-1,1) == labs.reshape(1,-1)).astype(np.float32)
        C[np.ix_(idx, idx)] += same
        I[np.ix_(idx, idx)] += 1.0

    with np.errstate(divide='ignore', invalid='ignore'):
        cons = np.zeros_like(C)
        mask = I > 0
        cons[mask] = C[mask] / I[mask]

    # PAC in (0.1, 0.9) on off-diagonal entries
    u_low, u_high = 0.1, 0.9
    offdiag = ~np.eye(n, dtype=bool)
    ambig = ((cons > u_low) & (cons < u_high) & offdiag).sum()
    total = offdiag.sum()
    pac = ambig / total
    return pac

def bootstrap_ari_stability(X, K, B=20, frac=0.8, random_state=0):
    """Fit on bootstrap resamples; predict labels for FULL X; pairwise ARIs across replicates."""
    rs = np.random.RandomState(random_state)
    X_np = _to_np(X)
    n = len(X_np)
    label_list = []
    for b in range(B):
        idx = rs.choice(n, size=max(2, int(frac*n)), replace=True)
        km = KMeans(n_clusters=K, n_init=10, max_iter=300, random_state=random_state+b).fit(X_np[idx])
        labels_full = km.predict(X_np)
        label_list.append(labels_full)
    aris = []
    for i in range(B):
        for j in range(i+1, B):
            aris.append(adjusted_rand_score(label_list[i], label_list[j]))
    return float(np.mean(aris)), float(np.std(aris))

def bootstrap_jaccard_stability_like_paper(X, K, B=100, frac=0.8, random_state=0):
    rs = np.random.RandomState(random_state)
    X_np = _to_np(X)
    n = len(X_np)
    km_full = KMeans(n_clusters=K, n_init=10, max_iter=300, random_state=random_state).fit(X_np)
    y_full = km_full.labels_

    j_scores = []
    for b in range(B):
        idx = rs.choice(n, size=max(2, int(frac*n)), replace=True)
        km_b = KMeans(n_clusters=K, n_init=10, max_iter=300, random_state=random_state+b+1).fit(X_np[idx])
        y_b = km_b.labels_
        j = jaccard_partition(y_full[idx], y_b)
        j_scores.append(j)
    return float(np.mean(j_scores)), float(np.std(j_scores))

def best_mapping_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    r, c = linear_sum_assignment(-cm)
    acc = cm[r, c].sum() / cm.sum()
    return float(acc)

def cross_source_ari(train_X, target_X, K, random_state=0):
    train_X = _to_np(train_X); target_X = _to_np(target_X)
    km_train = KMeans(n_clusters=K, n_init=10, max_iter=300, random_state=random_state).fit(train_X)
    transfer_labels = km_train.predict(target_X)
    km_native = KMeans(n_clusters=K, n_init=10, max_iter=300, random_state=random_state).fit(target_X)
    native_labels = km_native.labels_
    return adjusted_rand_score(transfer_labels, native_labels)

def tsne_plot(embeddings, labels, k, out_dir, name='internal', max_points=10000, dpi=110, seed=42):
    """t-SNE plotting using MulticoreTSNE if available, otherwise sklearn TSNE fallback."""
    try:
        from MulticoreTSNE import MulticoreTSNE as TSNE
    except Exception:
        from sklearn.manifold import TSNE

    rs = np.random.RandomState(seed)
    X_np = _to_np(embeddings)
    n = len(X_np)
    if n > max_points:
        idx = rs.choice(n, size=max_points, replace=False)
        X_use = X_np[idx]
        y_use = np.asarray(labels)[idx]
    else:
        X_use = X_np
        y_use = np.asarray(labels)

    tsne = TSNE(n_components=2, n_iter=300, random_state=seed)
    X_2d = tsne.fit_transform(X_use)
    plt.figure(figsize=(5,5), dpi=dpi)
    colors = sns.color_palette("deep", k)
    for cluster_id in range(k):
        m = (y_use == cluster_id) | (y_use == cluster_id + 1)
        pts = X_2d[m]
        if len(pts) == 0: continue
        plt.scatter(pts[:,0], pts[:,1], s=5, alpha=1.0, color=colors[cluster_id % len(colors)], label=f'C{cluster_id+1}')
    plt.title(f't-SNE for {k} clusters ({name})')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', markerscale=2.5)
    out = os.path.join(out_dir, f'tsne-{name}-{k}.pdf')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

# ----------------------------
# CLI arguments (no sensitive defaults)
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--disease", type=str, default='AD')
parser.add_argument("--model_name", type=str, default='cl_maskage_b32')
parser.add_argument("--stage", type=str, default='before')
parser.add_argument("--scale", type=int, default=1)
parser.add_argument("--k", type=int, default=5)
parser.add_argument("--fold_idx", type=int, default=2)
parser.add_argument("--seed", type=int, default=2024)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--device", type=str, default='cpu')

# Multi-K evaluation
parser.add_argument("--k_min", type=int, default=2)
parser.add_argument("--k_max", type=int, default=10)

# Gap statistic
parser.add_argument("--gap_B", type=int, default=10, help="Gap statistic reference draws per K")
parser.add_argument("--sil_sample", type=int, default=10000, help="Max samples for silhouette etc.")

# Consensus clustering (Monti PAC)
parser.add_argument("--consensus_reps", type=int, default=30)
parser.add_argument("--consensus_subsample", type=float, default=0.2)
parser.add_argument("--consensus_sample_size", type=int, default=1000, help="Cap N for consensus to keep NxN memory OK")

# Bootstrap stability (ARI and paper-style Jaccard)
parser.add_argument("--boot_B", type=int, default=10, help="Bootstrap replicates")
parser.add_argument("--boot_frac", type=float, default=0.2, help="Bootstrap sample fraction")

# Decision-tree replicability
parser.add_argument("--tree_max_depth", type=int, default=None)

# I/O for embeddings and results (no sensitive hard-coded paths)
parser.add_argument("--embedding_dir", type=str, default=".", help="Directory containing saved embeddings (optional)")
parser.add_argument("--x_train_file", type=str, default="", help="Path to saved X_train (.pt or .npy) - optional")
parser.add_argument("--cprd_x_file", type=str, default="", help="Path to saved CPRD_X_test (.pt or .npy) - optional")
parser.add_argument("--ukb_x_file", type=str, default="", help="Path to saved UKB_X_test (.pt or .npy) - optional")
parser.add_argument("--results_dir", type=str, default="./results", help="Directory to save results")

args = parser.parse_args()

# ----------------------------
# Prepare results directory
# ----------------------------
create_folder(args.results_dir)
results_over_k_dir = os.path.join(args.results_dir, "model_selection_EXTRA")
create_folder(results_over_k_dir)

# ----------------------------
# Load or accept embeddings
# ----------------------------
def load_embedding(path):
    if not path:
        return None
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embedding file not found: {path}")
    # Try torch load first, then numpy
    try:
        data = torch.load(path)
        return _to_np(data)
    except Exception:
        try:
            data = np.load(path, allow_pickle=True)
            return _to_np(data)
        except Exception as e:
            raise RuntimeError(f"Unable to load embedding from {path}: {e}")

# Priority: explicit file args > embedding_dir with filenames > in-memory variables if present
X_train = None; X_test = None; CPRD_X_test = None; UKB_X_test = None

# try explicit file args
if args.x_train_file:
    X_train = load_embedding(args.x_train_file)
if args.x_test_file:
    X_test = load_embedding(args.x_test_file)
if args.cprd_x_file:
    CPRD_X_test = load_embedding(args.cprd_x_file)
if args.ukb_x_file:
    UKB_X_test = load_embedding(args.ukb_x_file)

# try embedding_dir defaults (common filenames)
if X_train is None:
    candidate = os.path.join(args.embedding_dir, "eval_X_train.pt")
    if os.path.exists(candidate):
        X_train = load_embedding(candidate)
if CPRD_X_test is None:
    candidate = os.path.join(args.embedding_dir, "cprd_external_X_test.pt")
    if os.path.exists(candidate):
        CPRD_X_test = load_embedding(candidate)
if UKB_X_test is None:
    candidate = os.path.join(args.embedding_dir, "ukb_external_X_test.pt")
    if os.path.exists(candidate):
        UKB_X_test = load_embedding(candidate)

# As a last resort, check for in-memory variables (e.g., user previously defined them)
if X_train is None and "X_train" in globals():
    X_train = _to_np(globals()["X_train"])

if CPRD_X_test is None and "CPRD_X_test" in globals():
    CPRD_X_test = _to_np(globals()["CPRD_X_test"])
if UKB_X_test is None and "UKB_X_test" in globals():
    UKB_X_test = _to_np(globals()["UKB_X_test"])

# Require embeddings to proceed
missing = []
if X_train is None:
    missing.append("X_train")
if CPRD_X_test is None:
    missing.append("CPRD_X_test")
if UKB_X_test is None:
    missing.append("UKB_X_test")
if missing:
    raise RuntimeError(f"Missing required embeddings: {', '.join(missing)}. "
                       "Provide them via --x_train_file / --cprd_x_file / --ukb_x_file or set them in-memory.")

# X_test is optional for many downstream tasks; if missing, we will continue without it
if X_test is None:
    print("Warning: X_test not provided. Some evaluations that require internal test splits will be skipped or adapted.")

# Convert to numpy arrays for sklearn
X_tr_np   = _to_np(X_train)
X_cprd_np = _to_np(CPRD_X_test)
X_ukb_np  = _to_np(UKB_X_test)

# ----------------------------
# Multi-K reporting (K_min..K_max)
# ----------------------------
Ks = list(range(args.k_min, args.k_max + 1))
records = []
seed = args.seed
set_all_seeds(seed)

print(f"Evaluating K across {Ks} ...")
for K in Ks:
    print(f"[K={K}] metrics ...")
    # Derivation clustering & structure
    km_deriv = KMeans(n_clusters=K, n_init=10, max_iter=300, random_state=seed).fit(X_tr_np)
    y_tr = km_deriv.labels_
    sil_tr = silhouette_safe(X_tr_np, y_tr, max_n=args.sil_sample, random_state=seed)
    ch_tr  = calinski_harabasz_score(X_tr_np, y_tr)
    dbi_tr = davies_bouldin_score(X_tr_np, y_tr)

    # Gap statistic
    gaps, sks, _, _ = gap_statistic(X_tr_np, [K], B=args.gap_B, random_state=seed, max_n=args.sil_sample)
    gapK, skK = gaps[0], sks[0]

    # Consensus PAC
    pac = consensus_pac_monti(X_tr_np, K,
                              reps=args.consensus_reps,
                              subsample=args.consensus_subsample,
                              sample_size=args.consensus_sample_size,
                              random_state=seed)

    # Bootstrap ARI (stability)
    b_mean_ari, b_std_ari = bootstrap_ari_stability(X_tr_np, K, B=args.boot_B, frac=args.boot_frac, random_state=seed)

    # Cross-source ARIs (train derivation → assign target vs native target clusters)
    x_ari_cprd = cross_source_ari(X_tr_np, X_cprd_np, K, random_state=seed)
    x_ari_ukb  = cross_source_ari(X_tr_np, X_ukb_np,  K, random_state=seed)

    records.append({
        "K": K,
        "silhouette_train": sil_tr,
        "calinski_harabasz_train": ch_tr,
        "davies_bouldin_train": dbi_tr,
        "gap": gapK, "gap_se": skK,
        "consensus_PAC": pac,
        "bootstrap_ARI_mean": b_mean_ari,
        "bootstrap_ARI_std": b_std_ari,
        "cross_source_ARI_CPRDval": x_ari_cprd,
        "cross_source_ARI_UKB": x_ari_ukb,
    })

metrics_df = pd.DataFrame.from_records(records).sort_values("K")
metrics_csv = os.path.join(results_over_k_dir, f"model_selection_metrics_K{args.k_min}-{args.k_max}.csv")
metrics_df.to_csv(metrics_csv, index=False)
print(f"[saved] {metrics_csv}")

# Optionally: save final clustering for a chosen K (args.k)
chosen_K = args.k
km_final = KMeans(n_clusters=chosen_K, n_init=10, max_iter=300, random_state=seed).fit(X_tr_np)
labels_final = km_final.labels_
np.save(os.path.join(results_over_k_dir, f"clusters_K{chosen_K}.npy"), labels_final)
print(f"[saved] clusters for K={chosen_K}")

# Optionally: t-SNE plot for chosen K (if reasonably small)
try:
    tsne_plot(X_tr_np, labels_final, chosen_K, results_over_k_dir, name='derivation', max_points=10000, seed=seed)
    print("[saved] t-SNE plot")
except Exception as e:
    print(f"t-SNE generation failed: {e}")

print("Done.")
