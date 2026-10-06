import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d

# =============================
# Load data
# =============================
pkl_file = "fortest_To_compute_ZS_serial_score_per_file.pkl"
with open(pkl_file, "rb") as f:
    data = pickle.load(f)

# -----------------------------
# Normalize outer container
# -----------------------------
if isinstance(data, (list, tuple)):
    conf_iter = range(len(data))
    first_dict = data[0]
elif isinstance(data, dict):
    conf_iter = sorted(data.keys())
    first_dict = data[conf_iter[0]]
else:
    raise TypeError(type(data))

particles = sorted(first_dict.keys())
pid2idx = {pid: i for i, pid in enumerate(particles)}
N = len(particles)
M = len(conf_iter)

# -----------------------------
# Detect 1D / 2D
# -----------------------------
sample_val = next(iter(first_dict.values()))
has_D2 = (len(sample_val) == 3)
print(f"[INFO] Detected mode: {'2D' if has_D2 else '1D'}")

Z = np.empty((N, M))
D = np.empty((N, M))
D2 = np.empty((N, M)) if has_D2 else None

for j, ck in enumerate(conf_iter):
    dct = data[ck] if isinstance(data, dict) else data[ck]
    for pid, vals in dct.items():
        i = pid2idx[pid]
        if has_D2:
            z, dist, dist2 = vals
            Z[i, j] = z
            D[i, j] = dist
            D2[i, j] = dist2
        else:
            z, dist = vals
            Z[i, j] = z
            D[i, j] = dist

# =============================
# Histogram utilities
# =============================
def make_global_edges_1d(D, bins=50, clip_percentile=99.0):
    xmax = np.percentile(D, clip_percentile)
    xmin = min(0.0, np.percentile(D, 1.0))
    return np.linspace(xmin, xmax, bins + 1)

def make_global_edges_2d(D, D2, bins=(50, 50), clip_percentile=99.0):
    Nx, Ny = bins
    xmax = np.percentile(D, clip_percentile)
    ymax = np.percentile(D2, clip_percentile)
    xmin = min(0.0, np.percentile(D, 1.0))
    ymin = min(0.0, np.percentile(D2, 1.0))
    return (
        np.linspace(xmin, xmax, Nx + 1),
        np.linspace(ymin, ymax, Ny + 1),
    )

def z_landscape_1d(dist, Z, xedges, sigma=1.2):
    H, _ = np.histogram(dist, bins=xedges, weights=Z)
    C, _ = np.histogram(dist, bins=xedges)
    Zmean = H / np.maximum(C, 1)
    return gaussian_filter1d(Zmean, sigma=sigma), C

def z_landscape_2d(dist, dist2, Z, xedges, yedges, sigma=1.2):
    H, _, _ = np.histogram2d(dist, dist2, bins=[xedges, yedges], weights=Z)
    C, _, _ = np.histogram2d(dist, dist2, bins=[xedges, yedges])
    Zmean = H / np.maximum(C, 1)
    return gaussian_filter(Zmean, sigma=sigma), C

# =============================
# EB utilities
# =============================
def evidence_from_score_map(S, eps=1e-12, clip_cap=None):
    m = np.median(S)
    W = np.clip(S - m, 0.0, None)
    if clip_cap is not None:
        W = np.minimum(W, clip_cap)
    s = W.sum()
    if s <= 0:
        return np.full_like(W, 1.0 / W.size)
    return (W + eps) / (s + eps * W.size)

def kl_divergence(P, Q, eps=1e-12):
    P = np.clip(P, eps, 1.0)
    Q = np.clip(Q, eps, 1.0)
    return float(np.sum(P * (np.log(P) - np.log(Q))))

# =============================
# Empirical Bayes ranking
# =============================
def empirical_bayes_rank(
    Z, D, D2=None,
    bins=(50, 50),
    sigma_smooth=2.0,
    lam=2.0,
    max_iter=50,
    tol=1e-6,
    verbose=True,
):
    Np = Z.shape[0]

    # ---- grid ----
    if D2 is None:
        xedges = make_global_edges_1d(D, bins=bins[0])
        Nx = len(xedges) - 1
        S_maps = np.empty((Np, Nx))
        for i in range(Np):
            S, _ = z_landscape_1d(D[i], Z[i], xedges, sigma=sigma_smooth)
            S_maps[i] = S
    else:
        xedges, yedges = make_global_edges_2d(D, D2, bins=bins)
        Nx, Ny = bins
        S_maps = np.empty((Np, Nx, Ny))
        for i in range(Np):
            S, _ = z_landscape_2d(D[i], D2[i], Z[i], xedges, yedges, sigma=sigma_smooth)
            S_maps[i] = S

    cap = np.percentile(S_maps[S_maps > np.median(S_maps)], 99.5)

    P_list = np.array([evidence_from_score_map(S, clip_cap=cap) for S in S_maps])
    U = np.full_like(P_list[0], 1.0 / P_list[0].size)

    Pi = P_list.mean(axis=0)
    Pi /= Pi.sum()
    gamma = 0.5
    rho = np.full(Np, 0.5)

    for it in range(max_iter):
        ll1 = np.array([-lam * kl_divergence(P_list[i], Pi) for i in range(Np)])
        ll0 = np.array([-lam * kl_divergence(P_list[i], U) for i in range(Np)])

        loga = np.log(gamma + 1e-12) + ll1
        logb = np.log(1 - gamma + 1e-12) + ll0
        rho = 1 / (1 + np.exp(np.clip(logb - loga, -50, 50)))

        Pi_new = np.tensordot(rho, P_list, axes=(0, 0))
        Pi_new /= Pi_new.sum()
        gamma_new = rho.mean()

        if verbose:
            print(f"[EB] iter {it:02d} gamma={gamma_new:.4f}")

        if np.mean(np.abs(Pi - Pi_new)) < tol and abs(gamma - gamma_new) < tol:
            break

        Pi, gamma = Pi_new, gamma_new

#    return rho, Pi, dict(xedges=xedges, yedges=None if D2 is None else yedges)
    return rho, Pi, dict(xedges=xedges,yedges=None if D2 is None else yedges,P_list=P_list,S_maps=S_maps,U=U,)
# =============================
# Run
# =============================
rho, Pi, debug = empirical_bayes_rank(Z, D, D2)
def build_fraction_model(P_list, rho, fraction=0.1, mode="top", weights=None):
    """
    Build an empirical model from top or bottom fraction of particles by rho.

    Parameters
    ----------
    P_list : ndarray
        Evidence maps for all particles. Shape: (N, K) or (N, Kx, Ky)
    rho : ndarray
        Posterior signal probability for each particle.
    fraction : float
        Fraction of particles to use, e.g. 0.05, 0.10, 0.20.
    mode : str
        "top" or "bottom".
    weights : None or ndarray
        Optional particle weights. If None, simple average is used.

    Returns
    -------
    model : ndarray
        Averaged evidence model.
    selected_idx : ndarray
        Indices of selected particles.
    """
    Np = len(rho)
    n_select = max(1, int(round(Np * fraction)))

    if mode == "top":
        selected_idx = np.argsort(-rho)[:n_select]
    elif mode == "bottom":
        selected_idx = np.argsort(rho)[:n_select]
    else:
        raise ValueError("mode must be 'top' or 'bottom'")

    selected_P = P_list[selected_idx]

    if weights is None:
        model = selected_P.mean(axis=0)
    else:
        w = weights[selected_idx].astype(np.float64)
        w = w / np.maximum(w.sum(), 1e-12)
        model = np.tensordot(w, selected_P, axes=(0, 0))

    model = model / np.maximum(model.sum(), 1e-12)
    return model, selected_idx


def model_kl(A, B, eps=1e-12):
    A = np.clip(A, eps, 1.0)
    B = np.clip(B, eps, 1.0)
    return float(np.sum(A * (np.log(A) - np.log(B))))


def model_corr(A, B):
    a = A.ravel()
    b = B.ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom <= 0:
        return np.nan
    return float(np.sum(a * b) / denom)


def summarize_model(name, model, Pi, U):
    print(
        f"{name:12s} | "
        f"KL(model||Pi)={model_kl(model, Pi):.6f} | "
        f"KL(model||U)={model_kl(model, U):.6f} | "
        f"corr(model,Pi)={model_corr(model, Pi):.4f} | "
        f"entropy={-np.sum(np.clip(model, 1e-12, 1.0) * np.log(np.clip(model, 1e-12, 1.0))):.4f}"
    )


def plot_1d_models(models, debug, title="Top / bottom rho empirical models"):
    xedges = debug["xedges"]
    xc = 0.5 * (xedges[:-1] + xedges[1:])

    plt.figure(figsize=(8, 5))
    for name, model in models.items():
        plt.plot(xc, model, lw=2, label=name)

    plt.xlabel("Distance")
    plt.ylabel("Evidence probability")
    plt.title(title)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_1d_signal_background_difference(signal_model, bg_model, debug, title):
    xedges = debug["xedges"]
    xc = 0.5 * (xedges[:-1] + xedges[1:])

    diff = signal_model - bg_model

    plt.figure(figsize=(8, 4))
    plt.plot(xc, diff, lw=2)
    plt.axhline(0, color="gray", ls="--", lw=1)

    plt.xlabel("Distance")
    plt.ylabel("Signal model - background model")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_2d_model(model, debug, title):
    xedges = debug["xedges"]
    yedges = debug["yedges"]

    plt.figure(figsize=(5, 4))
    plt.imshow(
        model.T,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )
    plt.colorbar(label="Evidence probability")
    plt.xlabel("Rotation distance")
    plt.ylabel("Translation distance")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_2d_difference(signal_model, bg_model, debug, title):
    xedges = debug["xedges"]
    yedges = debug["yedges"]

    diff = signal_model - bg_model

    plt.figure(figsize=(5, 4))
    plt.imshow(
        diff.T,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )
    plt.colorbar(label="Signal - background")
    plt.xlabel("Rotation distance")
    plt.ylabel("Translation distance")
    plt.title(title)
    plt.tight_layout()
    plt.show()


P_list = debug["P_list"]
U = debug["U"]

# ---- Build models from rho-ranked particles ----
top05, idx_top05 = build_fraction_model(P_list, rho, fraction=0.05, mode="top")
top10, idx_top10 = build_fraction_model(P_list, rho, fraction=0.10, mode="top")
top20, idx_top20 = build_fraction_model(P_list, rho, fraction=0.20, mode="top")

bot10, idx_bot10 = build_fraction_model(P_list, rho, fraction=0.10, mode="bottom")
bot20, idx_bot20 = build_fraction_model(P_list, rho, fraction=0.20, mode="bottom")

models = {
    "EB Pi": Pi,
    "Uniform U": U,
    "Top 5%": top05,
    "Top 10%": top10,
    "Top 20%": top20,
    "Bottom 10%": bot10,
    "Bottom 20%": bot20,
}

# ---- Print rho ranges ----
print("\n[Diagnostic] rho ranges")
for name, idx in [
    ("Top 5%", idx_top05),
    ("Top 10%", idx_top10),
    ("Top 20%", idx_top20),
    ("Bottom 10%", idx_bot10),
    ("Bottom 20%", idx_bot20),
]:
    print(
        f"{name:12s}: N={len(idx):6d}, "
        f"rho min={rho[idx].min():.6f}, "
        f"rho max={rho[idx].max():.6f}, "
        f"rho mean={rho[idx].mean():.6f}"
    )

# ---- Print model similarity metrics ----
print("\n[Diagnostic] model similarity")
for name, model in models.items():
    summarize_model(name, model, Pi, U)

# ---- Plot models ----
if debug["yedges"] is None:
    plot_1d_models(models, debug)

    plot_1d_signal_background_difference(
        top05, bot20, debug,
        "Top 5% signal model - bottom 20% background model"
    )

    plot_1d_signal_background_difference(
        top10, bot20, debug,
        "Top 10% signal model - bottom 20% background model"
    )

    plot_1d_signal_background_difference(
        top20, bot20, debug,
        "Top 20% signal model - bottom 20% background model"
    )

else:
    for name, model in models.items():
        plot_2d_model(model, debug, name)

    plot_2d_difference(
        top05, bot20, debug,
        "Top 5% signal model - bottom 20% background model"
    )

    plot_2d_difference(
        top10, bot20, debug,
        "Top 10% signal model - bottom 20% background model"
    )

    plot_2d_difference(
        top20, bot20, debug,
        "Top 20% signal model - bottom 20% background model"
    )

def normalize_model(model, eps=1e-12):
    model = np.asarray(model, dtype=np.float64)
    s = model.sum()
    if s <= eps:
        return np.full_like(model, 1.0 / model.size)
    return model / s


def kl_divergence_map(P, Q, eps=1e-12):
    """
    KL(P || Q) for 1D or 2D evidence maps.
    """
    P = np.clip(P, eps, 1.0)
    Q = np.clip(Q, eps, 1.0)
    return float(np.sum(P * (np.log(P) - np.log(Q))))


def build_hard_fraction_model(P_list, rho, fraction=0.10, mode="top"):
    """
    Build empirical signal/background model using hard-cut rho ranking.

    Parameters
    ----------
    P_list : ndarray
        Evidence maps for all particles.
        Shape: (N, K) for 1D or (N, Kx, Ky) for 2D.
    rho : ndarray
        First-pass posterior signal probabilities.
    fraction : float
        Fraction of particles used to build model.
    mode : str
        "top" for high-rho signal model.
        "bottom" for low-rho background/noise model.

    Returns
    -------
    model : ndarray
        Normalized empirical model.
    selected_idx : ndarray
        Selected particle indices.
    """
    Np = len(rho)
    n_select = max(1, int(round(Np * fraction)))

    if mode == "top":
        selected_idx = np.argsort(-rho)[:n_select]
    elif mode == "bottom":
        selected_idx = np.argsort(rho)[:n_select]
    else:
        raise ValueError("mode must be 'top' or 'bottom'")

    model = P_list[selected_idx].mean(axis=0)
    model = normalize_model(model)

    return model, selected_idx


def second_pass_hard_cut(P_list, rho, signal_fraction=0.10, noise_fraction=0.10):
    """
    Second-pass scoring using hard-cut signal and empirical noise models.

    Signal model:
        average P_i from top signal_fraction rho particles

    Noise/background model:
        average P_i from bottom noise_fraction rho particles

    Score:
        delta_i = KL(P_i || B*) - KL(P_i || S*)

    Higher delta_i means more signal-like.
    """
    signal_model, signal_idx = build_hard_fraction_model(
        P_list, rho, fraction=signal_fraction, mode="top"
    )

    noise_model, noise_idx = build_hard_fraction_model(
        P_list, rho, fraction=noise_fraction, mode="bottom"
    )

    Np = len(rho)
    d_signal = np.empty(Np, dtype=np.float64)
    d_noise = np.empty(Np, dtype=np.float64)

    for i in range(Np):
        d_signal[i] = kl_divergence_map(P_list[i], signal_model)
        d_noise[i] = kl_divergence_map(P_list[i], noise_model)

    delta = d_noise - d_signal

    # A posterior-like score for convenience.
    # This is not a fully re-estimated EB posterior, just a monotonic transform of delta.
    delta_centered = delta - np.median(delta)
    delta_scale = np.std(delta_centered)

    if delta_scale > 1e-12:
        rho_second = 1.0 / (1.0 + np.exp(-delta_centered / delta_scale))
    else:
        rho_second = np.full_like(delta, 0.5)

    result = {
        "signal_model": signal_model,
        "noise_model": noise_model,
        "signal_idx": signal_idx,
        "noise_idx": noise_idx,
        "d_signal": d_signal,
        "d_noise": d_noise,
        "delta": delta,
        "rho_second": rho_second,
    }

    return result


# ---- Run second pass ----
P_list = debug["P_list"]

second = second_pass_hard_cut(
    P_list,
    rho,
    signal_fraction=0.10,
    noise_fraction=0.10,
)

signal_model = second["signal_model"]
noise_model = second["noise_model"]
delta = second["delta"]
rho_second = second["rho_second"]
# =============================
# Landscape matching classification
# =============================

def compute_evidence_strength_from_S_maps(S_maps, clip_cap=None, eps=1e-12):
    """
    Compute non-normalized evidence strength for each particle.

    This is different from P_list.
    P_list only keeps the shape of evidence after normalization.
    strength keeps the total amount of positive evidence.

    For each particle:
        m = median(S)
        W = max(S - m, 0)
        strength = sum(W)

    Parameters
    ----------
    S_maps : ndarray
        Smoothed score landscapes. Shape: (N, K) or (N, Kx, Ky)
    clip_cap : None or float
        Optional cap applied to W, consistent with evidence_from_score_map.
    eps : float

    Returns
    -------
    strength : ndarray
        Evidence strength for each particle.
    """
    Np = S_maps.shape[0]
    strength = np.empty(Np, dtype=np.float64)

    for i in range(Np):
        S = S_maps[i]
        m = np.median(S)
        W = np.clip(S - m, 0.0, None)

        if clip_cap is not None:
            W = np.minimum(W, clip_cap)

        strength[i] = W.sum()

    return strength


def classify_by_landscape_matching(
    P_list,
    class_models,
    background_model,
    evidence_strength=None,
    signal_delta_threshold=0.0,
    margin_threshold=0.0,
    strength_threshold=None,
    eps=1e-12,
):
    """
    Classify particles by matching each particle evidence map to
    class-specific landscape models and an empirical background model.

    For each particle i and class k:

        d_class[i,k] = KL(P_i || S_k)
        d_bg[i]      = KL(P_i || B)

        delta[i,k]   = d_bg[i] - d_class[i,k]

    Interpretation:
        delta > 0 means P_i is closer to class model than background.
        Larger delta means stronger class-like evidence.

    Assignment rule:
        1. If evidence strength is too low:
              label = "weak_background"
        2. Else find best class k* with max delta.
        3. If max_delta < signal_delta_threshold:
              label = "background"
        4. Else if margin to second-best class is too small:
              label = "ambiguous"
        5. Else:
              label = best class name

    Parameters
    ----------
    P_list : ndarray
        Evidence maps. Shape: (N, K) or (N, Kx, Ky).
    class_models : dict
        Dictionary of class models, e.g.
            {"class1": model1, "class2": model2}
        For your current test, use:
            {"signal": signal_model}
    background_model : ndarray
        Empirical background/noise model.
    evidence_strength : None or ndarray
        Non-normalized evidence strength for each particle.
    signal_delta_threshold : float
        Minimum KL contrast required to assign to a signal class.
        Default 0.0 means particle must be closer to class than background.
    margin_threshold : float
        Required gap between best and second-best class delta.
        For only one signal model, this has no effect.
    strength_threshold : None or float
        Minimum evidence strength required.
        If None, no strength filtering is applied.
    eps : float

    Returns
    -------
    result : dict
        Includes labels, best class, KL distances, delta matrix, margin, etc.
    """
    class_names = list(class_models.keys())
    models = [class_models[name] for name in class_names]

    Np = P_list.shape[0]
    K = len(models)

    d_bg = np.empty(Np, dtype=np.float64)
    d_class = np.empty((Np, K), dtype=np.float64)

    for i in range(Np):
        d_bg[i] = kl_divergence_map(P_list[i], background_model, eps=eps)
        for k in range(K):
            d_class[i, k] = kl_divergence_map(P_list[i], models[k], eps=eps)

    delta_mat = d_bg[:, None] - d_class

    best_k = np.argmax(delta_mat, axis=1)
    best_delta = delta_mat[np.arange(Np), best_k]
    best_class = np.array([class_names[k] for k in best_k], dtype=object)

    if K >= 2:
        sorted_delta = np.sort(delta_mat, axis=1)
        second_best_delta = sorted_delta[:, -2]
        margin = best_delta - second_best_delta
    else:
        second_best_delta = np.full(Np, np.nan)
        margin = np.full(Np, np.inf)

    labels = np.empty(Np, dtype=object)

    for i in range(Np):
        if strength_threshold is not None:
            if evidence_strength is None:
                raise ValueError("evidence_strength must be provided when strength_threshold is used.")

            if evidence_strength[i] < strength_threshold:
                labels[i] = "weak_background"
                continue

        if best_delta[i] < signal_delta_threshold:
            labels[i] = "background"
            continue

        if margin[i] < margin_threshold:
            labels[i] = "ambiguous"
            continue

        labels[i] = best_class[i]

    return {
        "labels": labels,
        "class_names": class_names,
        "best_class": best_class,
        "best_k": best_k,
        "d_background": d_bg,
        "d_class": d_class,
        "delta_mat": delta_mat,
        "best_delta": best_delta,
        "second_best_delta": second_best_delta,
        "margin": margin,
        "evidence_strength": evidence_strength,
        "signal_delta_threshold": signal_delta_threshold,
        "margin_threshold": margin_threshold,
        "strength_threshold": strength_threshold,
    }


def print_classification_summary(cls_result):
    labels = cls_result["labels"]
    unique, counts = np.unique(labels, return_counts=True)

    print("\n[Landscape matching classification summary]")
    for lab, cnt in zip(unique, counts):
        print(f"{lab:20s}: {cnt:8d}")

    print("\n[Score summary by assigned label]")
    for lab in unique:
        idx = np.where(labels == lab)[0]
        bd = cls_result["best_delta"][idx]
        print(
            f"{lab:20s}: "
            f"N={len(idx):8d}, "
            f"best_delta mean={bd.mean(): .6f}, "
            f"median={np.median(bd): .6f}, "
            f"min={bd.min(): .6f}, "
            f"max={bd.max(): .6f}"
        )


def plot_landscape_classification_scores(cls_result, rho=None, rho_second=None):
    labels = cls_result["labels"]
    best_delta = cls_result["best_delta"]

    order = np.argsort(best_delta)
    x = np.arange(len(best_delta))

    plt.figure(figsize=(7, 4))
    plt.scatter(x, best_delta[order], s=10, alpha=0.7)
    plt.axhline(cls_result["signal_delta_threshold"], color="gray", ls="--", lw=1)
    plt.xlabel("Particle rank by best landscape-matching delta")
    plt.ylabel("Best KL contrast")
    plt.title("Landscape matching score, sorted")
    plt.tight_layout()
    plt.show()

    if rho is not None:
        plt.figure(figsize=(6, 5))
        plt.scatter(rho, best_delta, s=10, alpha=0.7)
        plt.axhline(cls_result["signal_delta_threshold"], color="gray", ls="--", lw=1)
        plt.xlabel("First-pass rho")
        plt.ylabel("Best KL contrast")
        plt.title("First-pass rho vs landscape matching score")
        plt.tight_layout()
        plt.show()

    if rho_second is not None:
        plt.figure(figsize=(6, 5))
        plt.scatter(rho_second, best_delta, s=10, alpha=0.7)
        plt.axhline(cls_result["signal_delta_threshold"], color="gray", ls="--", lw=1)
        plt.xlabel("Second-pass posterior-like score")
        plt.ylabel("Best KL contrast")
        plt.title("Second-pass score vs landscape matching score")
        plt.tight_layout()
        plt.show()


def write_star_by_labels(
    labels,
    particles,
    input_star,
    output_prefix,
    keep_labels=None,
):
    """
    Write one STAR file for each requested label.

    Parameters
    ----------
    labels : ndarray
        Classification labels for particles.
    particles : list or ndarray
        particles[i] should be the original STAR file line number.
        This matches your current script logic.
    input_star : str
        Input STAR filename.
    output_prefix : str
        Prefix of output STAR files.
    keep_labels : None or list
        If None, write all labels.
        Otherwise only write labels in keep_labels.

    Returns
    -------
    written_files : dict
        label -> output filename
    """
    with open(input_star, "r") as f:
        lines = f.readlines()

    mline = find_particle_data_start(lines)

    if keep_labels is None:
        keep_labels = sorted(np.unique(labels))

    written_files = {}

    for lab in keep_labels:
        idx = np.where(labels == lab)[0]

        if len(idx) == 0:
            print(f"[STAR] Skip label {lab}: no particles")
            continue

        selected_SN = [particles[i] for i in idx]
        selected_SN = sorted(selected_SN)

        safe_lab = str(lab).replace("/", "_").replace(" ", "_")
        output_star = f"{output_prefix}_{safe_lab}.star"

        with open(output_star, "w") as fout:
            for i in range(mline):
                fout.write(lines[i])

            for SN in selected_SN:
                fout.write(lines[SN])

        written_files[lab] = output_star
        print(f"[STAR] Written {output_star}  N={len(selected_SN)}")

    return written_files
order_second = np.argsort(-delta)

print("\n[Second pass] hard-cut model summary")
print(f"Signal model: top 10% rho particles, N = {len(second['signal_idx'])}")
print(f"Noise model: bottom 10% rho particles, N = {len(second['noise_idx'])}")

print(
    f"Signal rho range: "
    f"{rho[second['signal_idx']].min():.6f} - "
    f"{rho[second['signal_idx']].max():.6f}"
)

print(
    f"Noise rho range: "
    f"{rho[second['noise_idx']].min():.6f} - "
    f"{rho[second['noise_idx']].max():.6f}"
)

print("\n[Second pass] delta summary")
print(f"delta min    = {delta.min():.6f}")
print(f"delta max    = {delta.max():.6f}")
print(f"delta mean   = {delta.mean():.6f}")
print(f"delta median = {np.median(delta):.6f}")
print(f"delta std    = {delta.std():.6f}")

print("\nTop 10 particles by second-pass delta:")
for i in order_second[:10]:
    print(
        f"particle={particles[i]}  "
        f"rho1={rho[i]:.6f}  "
        f"delta={delta[i]:.6f}  "
        f"rho2_like={rho_second[i]:.6f}  "
        f"d_signal={second['d_signal'][i]:.6f}  "
        f"d_noise={second['d_noise'][i]:.6f}"
    )

print("\nBottom 10 particles by second-pass delta:")
for i in order_second[-10:]:
    print(
        f"particle={particles[i]}  "
        f"rho1={rho[i]:.6f}  "
        f"delta={delta[i]:.6f}  "
        f"rho2_like={rho_second[i]:.6f}  "
        f"d_signal={second['d_signal'][i]:.6f}  "
        f"d_noise={second['d_noise'][i]:.6f}"
    )
# =============================
# Plot second-pass models and scores
# =============================

def plot_second_pass_models_1d(signal_model, noise_model, Pi, U, debug):
    xedges = debug["xedges"]
    xc = 0.5 * (xedges[:-1] + xedges[1:])

    plt.figure(figsize=(8, 5))

    plt.plot(xc, signal_model, lw=2, label="Second-pass signal model, top 10%")
    plt.plot(xc, noise_model, lw=2, label="Second-pass background model, bottom 10%")
    plt.plot(xc, Pi, lw=2, ls="--", label="First-pass EB Pi")
    plt.plot(xc, U, lw=1.5, ls=":", label="Uniform U")

    plt.xlabel("Distance")
    plt.ylabel("Evidence probability")
    plt.title("Second-pass empirical signal/background models")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_second_pass_difference_1d(signal_model, noise_model, debug):
    xedges = debug["xedges"]
    xc = 0.5 * (xedges[:-1] + xedges[1:])

    diff = signal_model - noise_model

    plt.figure(figsize=(8, 4))
    plt.plot(xc, diff, lw=2)
    plt.axhline(0, color="gray", ls="--", lw=1)

    plt.xlabel("Distance")
    plt.ylabel("Signal model - background model")
    plt.title("Second-pass model contrast")
    plt.tight_layout()
    plt.show()


def plot_second_pass_scores(rho, delta, rho_second):
    plt.figure(figsize=(6, 5))
    plt.scatter(rho, delta, s=10, alpha=0.7)

    plt.xlabel("First-pass posterior rho")
    plt.ylabel("Second-pass KL contrast delta")
    plt.title("First-pass rho vs second-pass delta")
    plt.tight_layout()
    plt.show()

    order = np.argsort(delta)
    x = np.arange(len(delta))

    plt.figure(figsize=(7, 4))
    plt.scatter(x, delta[order], s=10, alpha=0.7)
    plt.axhline(0, color="gray", ls="--", lw=1)

    plt.xlabel("Particle rank by second-pass delta")
    plt.ylabel("Second-pass KL contrast delta")
    plt.title("Second-pass delta, sorted")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.scatter(x, rho_second[order], s=10, alpha=0.7)
    plt.axhline(0.5, color="gray", ls="--", lw=1)

    plt.xlabel("Particle rank by second-pass delta")
    plt.ylabel("Posterior-like second-pass score")
    plt.title("Second-pass posterior-like score, sorted")
    plt.tight_layout()
    plt.show()


def plot_model_2d(model, debug, title, colorbar_label="Evidence probability"):
    xedges = debug["xedges"]
    yedges = debug["yedges"]

    plt.figure(figsize=(5, 4))
    plt.imshow(
        model.T,
        origin="lower",
        aspect="auto",
        extent=[
            xedges[0], xedges[-1],
            yedges[0], yedges[-1],
        ],
    )
    plt.colorbar(label=colorbar_label)
    plt.xlabel("Rotation distance")
    plt.ylabel("Translation distance")
    plt.title(title)
    plt.tight_layout()
    plt.show()


if debug["yedges"] is None:
    plot_second_pass_models_1d(
        signal_model,
        noise_model,
        Pi,
        debug["U"],
        debug,
    )

    plot_second_pass_difference_1d(
        signal_model,
        noise_model,
        debug,
    )

else:
    plot_model_2d(
        signal_model,
        debug,
        "Second-pass signal model, top 10%",
    )

    plot_model_2d(
        noise_model,
        debug,
        "Second-pass background model, bottom 10%",
    )

    plot_model_2d(
        signal_model - noise_model,
        debug,
        "Second-pass signal - background",
        colorbar_label="Signal - background",
    )

plot_second_pass_scores(rho, delta, rho_second)
order = np.argsort(-rho)
#order_second = np.argsort(-delta)

print("Top 10:")
for i in order[:10]:
    print(particles[i], rho[i])

print("Bottom 10:")
for i in order[-10:]:
    print(particles[i], rho[i])

# =============================
# Plot
# =============================
def plot_particle(i):
    if D2 is None:
        S, _ = z_landscape_1d(D[i], Z[i], debug["xedges"])
        xc = 0.5 * (debug["xedges"][:-1] + debug["xedges"][1:])
        plt.plot(xc, S)
        plt.axhline(0, color="gray", ls="--")
        plt.xlabel("Distance")
        plt.ylabel("Smoothed Z")
    else:
        S, _ = z_landscape_2d(D[i], D2[i], Z[i], debug["xedges"], debug["yedges"])
        plt.imshow(
            S.T,
            origin="lower",
            aspect="auto",
            extent=[
                debug["xedges"][0], debug["xedges"][-1],
                debug["yedges"][0], debug["yedges"][-1],
            ],
        )
        plt.colorbar(label="Smoothed Z")
        plt.xlabel("Rotation distance")
        plt.ylabel("Translation distance")

    plt.title(f"Particle {particles[i]}")
    plt.tight_layout()
    plt.show()

#for i in order[:5]:
#    plot_particle(i)
#for i in order[-5:]:
#    plot_particle(i)

# =============================
# Plot rho for all particles
# =============================
plt.figure(figsize=(6, 4))

x = np.arange(len(rho))
y = rho

plt.scatter(x, y, s=10, alpha=0.8)
plt.axhline(0.5, color="gray", ls="--", lw=1)

plt.xlabel("Particle index (sorted by input order)")
plt.ylabel("Posterior signal probability ρ")

plt.title("Empirical Bayes posterior per particle")
plt.tight_layout()
plt.show()
# =============================
# Plot rho (sorted)
# =============================
plt.figure(figsize=(6, 4))

order = np.argsort(rho)
y = rho[order]
x = np.arange(len(y))

plt.scatter(x, y, s=10, alpha=0.8)
plt.axhline(0.5, color="gray", ls="--", lw=1)

plt.xlabel("Particle rank (low → high ρ)")
plt.ylabel("Posterior signal probability ρ")

plt.title("Empirical Bayes posterior per particle (sorted)")
plt.tight_layout()
plt.show()

def find_particle_data_start(lines):
    particle_block_candidates = ("data_particles", "data_images")  # data_images for some variants
    start_block = None
    for i, line in enumerate(lines):
        s = line.strip()
        if any(s.startswith(name) for name in particle_block_candidates):
            start_block = i
            break

    if start_block is None:
        raise RuntimeError("Cannot find particles block (expected 'data_particles' or 'data_images').")

    # 2) Inside particles block, find the loop_ that defines the particles table
    loop_line = None
    for i in range(start_block, len(lines)):
        s = lines[i].strip()
        # stop if another data_ block starts before we found loop_ (unexpected)
        if i != start_block and s.startswith("data_"):
            break
        if s.startswith("loop_"):
            loop_line = i
            break

    if loop_line is None:
        raise RuntimeError("Cannot find 'loop_' inside the particles block.")

    # 3) After loop_, skip column-name lines that start with '_' (and blanks/comments),
    #    the first "real" row is particle data start.
    for i in range(loop_line + 1, len(lines)):
        s = lines[i].strip()
        if s == "" or s.startswith("#"):
            continue
        if s.startswith("_"):
            continue
        # First non-header row => particle data begins
        return i

    raise RuntimeError("Reached EOF without finding particle data rows.")
    
# =============================
# Run landscape matching classifier
# current version: one signal model + one empirical background model
# =============================

class_models = {
    "signal": signal_model,
}

# Optional evidence-strength filter.
# This uses original non-normalized positive evidence.
# If you do not want this filter, set strength_threshold = None.
evidence_strength = compute_evidence_strength_from_S_maps(
    debug["S_maps"],
    clip_cap=None,
)

# 一个比较保守的起点：
# 只过滤掉 evidence strength 最低的 5%。
# 如果不想用 strength filter，可以设成 None。
strength_threshold = np.percentile(evidence_strength, 5.0)

cls = classify_by_landscape_matching(
    P_list=P_list,
    class_models=class_models,
    background_model=noise_model,
    evidence_strength=evidence_strength,
    signal_delta_threshold=0.0,
    margin_threshold=0.0,
    strength_threshold=strength_threshold,
)

print_classification_summary(cls)
plot_landscape_classification_scores(cls, rho=rho, rho_second=rho_second)


####
input_star = "input.star"
output_star = "rho_low_2pass_25k.star"
N_select = 25000   # 从小到大取前 1400

with open(input_star, "r") as f:
    lines = f.readlines()

mline = find_particle_data_start(lines)

# rho 从小到大排序
order = np.argsort(rho)

# 取前 N_select 个
#selected_idx = order[:N_select]
# Now use the second pass result
order_second = np.argsort(rho_second)
selected_idx = order_second[:N_select]

# 对应 star 行号（particle[i] 就是行号）
selected_SN = [particles[i] for i in selected_idx]

# ⚠️ 强烈建议排序一下行号，保持 star 原始顺序
selected_SN = sorted(selected_SN)

with open(output_star, "w") as fout:
    # 1️⃣ header 原样写
    for i in range(mline):
        fout.write(lines[i])

    # 2️⃣ 写 particle 行
    for SN in selected_SN:
        fout.write(lines[SN])

print(f"Written: {output_star}  (N={len(selected_SN)})")
written = write_star_by_labels(
    labels=cls["labels"],
    particles=particles,
    input_star="input.star",
    output_prefix="landscape_match",
    keep_labels=["signal", "background", "weak_background"],
)