# %%
from Customize import JAVA_PATH, CPRD_CODE_PATH, COHORT_SAVE_PATH, CBEHRT_PATH, MODEL_SAVE_PATH
import warnings
warnings.filterwarnings("ignore")
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys 
sys.path.insert(0, CBEHRT_PATH)
sys.path.insert(0, CPRD_CODE_PATH)
import pandas as pd
import numpy as np
import random
import _pickle as pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, fowlkes_mallows_score, adjusted_rand_score, calinski_harabasz_score
from lifelines import CoxPHFitter
from MulticoreTSNE import MulticoreTSNE as TSNE
from tqdm import tqdm
from joblib import parallel_backend

# %%
def create_folder(path):
    os.makedirs(path, exist_ok=True)
# ehr_data_path = '/home/zhengxian/HF_pheno/ehr_data/'
import warnings
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--disease", type=str, default='AD')
parser.add_argument("--cohort_dir", type=str, default='AD_data')
parser.add_argument("--experient_dir", type=str, default='AD')
parser.add_argument("--model_name", type=str, default='TFIDF_KMeans')
parser.add_argument("--stage", type=str, default='before')
parser.add_argument("--seed", type=int, default=2024)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--device", type=str, default='0')
# Evaluation extras with safe defaults (keep your flow; change model only)
parser.add_argument("--consensus_resamples", type=int, default=100)
parser.add_argument("--bootstrap_runs", type=int, default=100)
# parser.add_argument("--gap_b", type=int, default=50)
parser.add_argument("--subsample_frac", type=float, default=0.8)
args = parser.parse_args()

seed = args.seed
stage = args.stage
cohort_path = os.path.join(COHORT_SAVE_PATH, args.cohort_dir)
ehr_data_path = os.path.join(cohort_path, 'EHR')
experient_dir = os.path.join(MODEL_SAVE_PATH, args.experient_dir)
create_folder(experient_dir)
model_save_dir = os.path.join(experient_dir, args.model_name +'_'+stage)
create_folder(model_save_dir)
results_dir = os.path.join(model_save_dir, 'results')
create_folder(results_dir)

# %%
import math
def set_all_seeds(seed):
    """Set all seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)

    
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def normalise_tokens_col(series):
    # expect a list/array of codes; if strings, split on whitespace/commas
    def _to_text(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return " ".join(map(str, x))
        x = "" if pd.isna(x) else str(x)
        x = x.replace(",", " ")
        return " ".join(x.split())
    return series.apply(_to_text).tolist()


def within_cluster_dispersion(X, labels, centers):
    # X dense expected
    diffs = X - centers[labels]
    return float(np.sum(diffs**2))




def consensus_pac(X_dense, k, resamples=100, subsample_frac=0.8, random_state=seed):
    rs = np.random.RandomState(random_state)
    n = X_dense.shape[0]
    C = np.zeros((n, n), dtype=np.float32)
    counts = np.zeros((n, n), dtype=np.int32)
    for _ in range(resamples):
        idx = rs.choice(n, size=int(math.ceil(subsample_frac * n)), replace=False)
        km = KMeans(n_clusters=k, n_init=10, random_state=rs.randint(0, 10**9))
        labels = km.fit_predict(X_dense[idx])
        for a in range(len(idx)):
            for b in range(a+1, len(idx)):
                i, j = idx[a], idx[b]
                same = 1 if labels[a] == labels[b] else 0
                C[i, j] += same; C[j, i] += same
                counts[i, j] += 1; counts[j, i] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        M = np.divide(C, counts, out=np.zeros_like(C), where=counts>0)
    mask = (M > 0.1) & (M < 0.9)
    denom = np.sum(counts > 0) - n
    pac = float(np.sum(mask)) / float(max(denom, 1))
    return pac


def bootstrap_ari(X_dense, k, runs=100, random_state=42):
    rng = np.random.RandomState(random_state)
    n = X_dense.shape[0]
    labelings = []
    for _ in range(runs):
        idx = rng.choice(n, size=n, replace=True)
        km = KMeans(n_clusters=k, n_init=10, random_state=rng.randint(0, 10**9))
        km.fit(X_dense[idx])
        centers = km.cluster_centers_
        dists = np.linalg.norm(X_dense[:, None, :] - centers[None, :, :], axis=2)
        labels_full = np.argmin(dists, axis=1)
        labelings.append(labels_full)
    if len(labelings) < 2:
        return 1.0, 0.0
    aris = []
    for i in range(len(labelings)):
        for j in range(i+1, len(labelings)):
            aris.append(adjusted_rand_score(labelings[i], labelings[j]))
    return float(np.mean(aris)), float(np.std(aris, ddof=1))


def assign_by_centers(X_dense, centers):
    dists = np.linalg.norm(X_dense[:, None, :] - centers[None, :, :], axis=2)
    return np.argmin(dists, axis=1)


def km_plot(time, event, labels, out_png, title):
    if not LIFELINES_AVAILABLE:
        return False
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(6,4))
    for k in np.unique(labels):
        m = labels == k
        if m.sum() == 0: 
            continue
        kmf.fit(time[m], event_observed=event[m], label=f"Cluster {k+1}")
        kmf.plot()
    plt.title(title); plt.xlabel("Time"); plt.ylabel("Survival probability")
    plt.tight_layout(); plt.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close()
    return True

def calculate_prediction_strength(test_set, test_labels, training_centers):
    # Precompute the distances for all points to training_centers
    distances = np.linalg.norm(test_set[:, np.newaxis] - training_centers, axis=2)
    closest_centers = np.argmin(distances, axis=1)
    
    # For each cluster in test_labels, compute the prediction strength
    prediction_strengths = []
    for cluster in range(training_centers.shape[0]):
        indices = np.where(test_labels == cluster)[0]
        cluster_size = len(indices)
        
        if cluster_size <= 1:
            prediction_strengths.append(float('inf'))
        else:
            # Get the closest centers for all points in this cluster
            cluster_closest_centers = closest_centers[indices]
            
            # Calculate how many pairs have the same closest center
            matching_pairs = np.sum(np.triu((cluster_closest_centers[:, None] == cluster_closest_centers), 1))
            total_pairs = cluster_size * (cluster_size - 1) / 2.0
            prediction_strengths.append(matching_pairs / total_pairs)
            
    return min(prediction_strengths)

# The rest of the code remains largely the same, but you can remove the `closest_center` and `prediction_strength_of_cluster` functions.



# %% [markdown]
# # Read data

# %%
ehr_df_internal = pd.read_parquet(os.path.join(ehr_data_path, f'ehr_b4_{args.disease}_internal'))
ehr_df_external = pd.read_parquet(os.path.join(ehr_data_path, f'ehr_b4_{args.disease}_external'))

# %%
# build corpus and initialise vectorizer
m_path = ''
mp = load_obj(m_path)['token2idx']
ehr_df_internal.code = ehr_df_internal.code.apply(lambda x: ' '.join([each for each in x if each != 'SEP' and mp.get(each, 'UNK') != 'UNK'])).values
ehr_df_external.code = ehr_df_external.code.apply(lambda x: ' '.join([each for each in x if each != 'SEP' and mp.get(each, 'UNK') != 'UNK'])).values

# %%


set_all_seeds(seed)
all_metrics = []
ps_threshold = 0.95

print(len(ehr_df_internal))
kf = KFold(n_splits=5, shuffle=True)

cluster_range = range(2, 10)
for fold_idx, (train_idx, test_idx) in enumerate(tqdm(kf.split(ehr_df_internal), desc="KFold splits")):
    print(f"\n--- Fold {fold_idx+1} ---")

    # === Vectorize and reduce ===
    vectorizer = TfidfVectorizer(vocabulary=(list(mp.keys())[5:]), lowercase=False)
    X_train = vectorizer.fit_transform(ehr_df_internal.code.iloc[train_idx])
    X_test = vectorizer.transform(ehr_df_internal.code.iloc[test_idx])
    external_X = vectorizer.transform(ehr_df_external.code)

    svd = TruncatedSVD(n_components=300, random_state=seed)
    X_train = svd.fit_transform(X_train)
    X_test = svd.transform(X_test)
    external_X = svd.transform(external_X)
    
    silhouette_k, ps_k, ch_k, db_k = ([] for _ in range(4))

    for k in tqdm(cluster_range, desc=f'Fold {fold_idx+1} - number of clusters'):
        print(f"\n=== Evaluating k={k} ===")
        # step_start = time.time()

        # --- Training & label assignment ---
        # t0 = time.time()
        km_tr = KMeans(n_clusters=k, max_iter=50, n_init=10, random_state=seed).fit(X_train)
        km_te = KMeans(n_clusters=k, max_iter=50, n_init=10, random_state=seed).fit(X_test)
        train_labels = km_tr.predict(X_test)
        # print(f"  [Time] KMeans fit+predict: {time.time() - t0:.2f}s")
        external_train_labels = km_tr.predict(external_X)


        # --- Core metrics ---
        # t0 = time.time()
        sil = float(silhouette_score(X_test, train_labels))
        ps  = calculate_prediction_strength(X_test, km_te.labels_, km_tr.cluster_centers_)
        ch  = float(calinski_harabasz_score(X_test, train_labels))
        db  = float(davies_bouldin_score(X_test, train_labels))
        # print(f"  [Time] Core metrics: {time.time() - t0:.2f}s")

        # step_time = time.time() - step_start
        # print(f"✅ Total time for k={k}: {step_time:.2f}s")

        silhouette_k.append(sil)
        ps_k.append(ps)
        ch_k.append(ch)
        db_k.append(db)

    # === Save fold results ===
    for i, k in enumerate(cluster_range):
        all_metrics.append({
            "fold": fold_idx + 1,
            "k": k,
            "silhouette": silhouette_k[i],
            "prediction_strength": ps_k[i],
            "calinski_harabasz": ch_k[i],
            "davies_bouldin": db_k[i],
        })

# === Aggregate metrics ===
df_metrics = pd.DataFrame(all_metrics)
df_summary = df_metrics.groupby("k").agg({
    "silhouette": "mean",
    "prediction_strength": "mean",
    "calinski_harabasz": "mean",
    "davies_bouldin": "mean"
}).reset_index()

# # === Select best k (PS threshold, fallback to highest silhouette) ===
valid_k = df_summary[df_summary["prediction_strength"] >= ps_threshold]
if not valid_k.empty:
    best_k = int(valid_k.iloc[-1]["k"])
else:
    best_k = int(df_summary.iloc[df_summary["silhouette"].idxmax()]["k"])

print(f"\n✅ Optimal k selected: {best_k}")
print(df_summary.loc[df_summary['k'] == best_k])
# === Save metrics ===
df_metrics.to_csv(os.path.join(results_dir, "tfidf_metrics_across_k.csv"), index=False)
df_summary.to_csv(os.path.join(results_dir, "tfidf_metrics_summary.csv"), index=False)


optimal_k_ps = best_k
ps_scores = df_summary["prediction_strength"].tolist()
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(cluster_range, ps_scores, marker='o', linestyle='-', color='b', label='Prediction Strength')
plt.axvline(x=optimal_k_ps, color='r', linestyle='--', label=f'Optimal k = {optimal_k_ps}')
plt.axhline(y=ps_threshold, color='g', linestyle='--', label=f'Threshold = {ps_threshold}')
plt.title('Prediction Strength for Different k Values')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Prediction Strength')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "tfidf_prediction_strength_across_k.png"), dpi=300, bbox_inches="tight")
# plt.show()

# %%
def tsne_plot(embeddings, labels, k, dataset = 'internal'):
    tsne = TSNE(n_components=2, n_iter=300, random_state=seed)
    X_test_2D = tsne.fit_transform(embeddings[:10000])
    plt.figure(figsize=(5,5), dpi=300)
    colors = sns.color_palette("deep", k) 
    
    for cluster_id in range(1, k+1):
        # Separate the data points by cluster
        cluster_mask = (labels[:10000] == cluster_id)
        cluster_data = X_test_2D[cluster_mask]
        
        # Plot the cluster points and label for the legend
        plt.scatter(
            cluster_data[:, 0], cluster_data[:, 1],
            alpha=1.0, color=colors[cluster_id-1],
            s = 5,
            label=f'Cluster {cluster_id}' 
        )

    plt.title(f't-SNE visualization for {k} clusters')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large', markerscale=2.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'tsne_k{optimal_k_ps}_{dataset}.png'), dpi=300, bbox_inches="tight")
    plt.show()


kf = KFold(n_splits=5, shuffle=True)
best_fold_idx = 2
optimal_k_ps = best_k

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(ehr_df_internal)):
    if fold_idx != best_fold_idx:
        continue

    vectorizer = TfidfVectorizer(vocabulary=(list(mp.keys())[5:]), lowercase=False)
    X_train = vectorizer.fit_transform(ehr_df_internal.code.iloc[train_idx])
    X_test = vectorizer.transform(ehr_df_internal.code.iloc[test_idx])
    external_X = vectorizer.transform(ehr_df_external.code)

    svd = TruncatedSVD(n_components=300, random_state=seed)
    X_train = svd.fit_transform(X_train)
    X_test = svd.transform(X_test)
    external_X = svd.transform(external_X)

    # Fit KMeans on training fold
    km_tr = KMeans(n_clusters=optimal_k_ps, max_iter=50, n_init=10, random_state=seed).fit(X_train)
    # Predict labels
    km_te = KMeans(n_clusters=optimal_k_ps, max_iter=50, n_init=10, random_state=seed).fit(X_test)
    
    train_labels = km_tr.predict(X_test) + 1
    external_train_labels = km_tr.predict(external_X) + 1

    print(f"✅ Fold {fold_idx+1} done. Shapes: X_train={X_train.shape}, X_test={X_test.shape}, external_X={external_X.shape}")

# %%
set_all_seeds(seed)
print('internal validation')
tsne_plot(X_test, train_labels, optimal_k_ps, dataset= 'internal')

print('external validation')
tsne_plot(external_X, external_train_labels, optimal_k_ps, dataset= 'external')


# %%
metrics_records = []

# === Internal validation ===
print("\n=== Internal validation ===")
t0 = time.time()
tsne_plot(X_test, train_labels, optimal_k_ps, dataset='internal')

# --- Core metrics ---
t1 = time.time()
sil = float(silhouette_score(X_test, train_labels))
# ps  = calculate_prediction_strength(X_test, km_te.labels_, km_tr.cluster_centers_)
ch  = float(calinski_harabasz_score(X_test, train_labels))
db  = float(davies_bouldin_score(X_test, train_labels))
core_time = time.time() - t1
print(f"[Core] Sil={sil:.4f},  CH={ch:.2f}, DB={db:.2f} ({core_time:.2f}s)")

# --- Advanced metrics ---
t2 = time.time()
pac = consensus_pac(X_test, optimal_k_ps, resamples=5, subsample_frac=0.2)
b_mean, b_sd = bootstrap_ari(X_test, optimal_k_ps, runs=5, random_state=seed)
adv_time = time.time() - t2
print(f"[Advanced] PAC={pac:.4f}, BootARI={b_mean:.4f}±{b_sd:.4f} ({adv_time:.2f}s)")

# internal_time = time.time() - t0
# print(f"✅ Internal validation done in {internal_time:.2f}s")

metrics_records.append({
    "dataset": "internal",
    "k": optimal_k_ps,
    "silhouette": sil,
    # "prediction_strength": ps,
    "calinski_harabasz": ch,
    "davies_bouldin": db,
    "consensus_pac": pac,
    "bootstrap_ari_mean": b_mean,
    "bootstrap_ari_sd": b_sd,
    "cross_source_ari_validation": None,
    "time_core": core_time,
    "time_advanced": adv_time,
    # "time_total": internal_time
})

# === External validation ===
print("\n=== External validation ===")
t0 = time.time()
tsne_plot(external_X, external_train_labels, optimal_k_ps, dataset='external')
# --- Core metrics ---
t1 = time.time()
sil_ext = float(silhouette_score(external_X, external_train_labels))
ch_ext  = float(calinski_harabasz_score(external_X, external_train_labels))
db_ext  = float(davies_bouldin_score(external_X, external_train_labels))
core_time_ext = time.time() - t1
print(f"[Core] Sil={sil_ext:.4f}, CH={ch_ext:.2f}, DB={db_ext:.2f} ({core_time_ext:.2f}s)")

# --- Advanced metrics ---
t2 = time.time()
pac_ext = consensus_pac(external_X, optimal_k_ps, resamples=5, subsample_frac=0.2)
b_mean_ext, b_sd_ext = bootstrap_ari(external_X, optimal_k_ps, runs=5, random_state=seed)
adv_time_ext = time.time() - t2
print(f"[Advanced] PAC={pac_ext:.4f}, BootARI={b_mean_ext:.4f}±{b_sd_ext:.4f} ({adv_time_ext:.2f}s)")

# --- Cross-source ARI ---
t3 = time.time()
km_internal = KMeans(n_clusters=optimal_k_ps, n_init=10, random_state=seed).fit(X_test)
km_external = KMeans(n_clusters=optimal_k_ps, n_init=10, random_state=seed).fit(external_X)
# labels_external_assigned = assign_by_centers(fit, km_internal.cluster_centers_)
labels_external_assigned = assign_by_centers(external_X, km_internal.cluster_centers_)
cross_ari = float(adjusted_rand_score(km_external.labels_, labels_external_assigned))
cross_time = time.time() - t3
print(f"[Cross-source] ARI={cross_ari:.4f} ({cross_time:.2f}s)")

external_time = time.time() - t0
print(f"✅ External validation done in {external_time:.2f}s")

metrics_records.append({
    "dataset": "external",
    "k": optimal_k_ps,
    "silhouette": sil_ext,
    # "prediction_strength": ps_ext,
    "calinski_harabasz": ch_ext,
    "davies_bouldin": db_ext,
    "consensus_pac": pac_ext,
    "bootstrap_ari_mean": b_mean_ext,
    "bootstrap_ari_sd": b_sd_ext,
    "cross_source_ari_validation": cross_ari,
    "time_core": core_time_ext,
    "time_advanced": adv_time_ext,
    "time_total": external_time
})

# === Combine & Save ===
df_metrics = pd.DataFrame(metrics_records)
df_metrics.to_csv(os.path.join(results_dir, "tfidf_validation_metrics.csv"), index=False)