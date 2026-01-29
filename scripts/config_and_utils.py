# config_and_utils.py
import numpy as np


def sublinear_tau_fn(b, alpha=0.2, beta=0.1, gamma=0.7):
    """
    Example tau_k(b). You can replace with real measured profile later.
    Must return time in the same unit as arrival timestamps.
    """
    b = int(max(1, b))
    return float(beta + alpha * (b ** gamma))


# -------------------- Profile-based tau builder (NEW) --------------------

def normalize_gpu_name(gpu: str) -> str:
    """Map verbose GPU names to canonical short names used in experiments."""
    s = str(gpu).upper()
    if "A100" in s:
        return "A100"
    if "H100" in s:
        return "H100"
    if "L40S" in s or "L40" in s:
        return "L40S"
    return str(gpu)


def normalize_pipeline_name(pipeline: str, model: str = "") -> str:
    """
    Normalize pipeline names for matching.
    - If pipeline is 'sdxl' and model contains 'sdxl-turbo', treat as 'sdxl_turbo'
    - Normalize sd1.5 variants to 'sd15'
    """
    p = str(pipeline).lower().strip()
    m = str(model).lower().strip()

    if p == "sdxl" and "sdxl-turbo" in m:
        return "sdxl_turbo"

    if p in ["sd15", "sd1.5", "sd_1_5", "stable-diffusion-v1-5", "stable_diffusion_v1_5"]:
        return "sd15"

    return p


def build_tau_from_profile_csv(
    csv_path: str,
    gpu: str,
    pipeline: str,
    steps: int,
    batch_max: int,
    *,
    latency_per_image_col: str = "latency_per_image_s",
    batch_col: str = "batch",
    gpu_col: str = "gpu",
    pipeline_col: str = "pipeline",
    steps_col: str = "steps",
):
    """
    Build tau(b) (batch busy time) from an empirical profile CSV.

    Expected CSV columns (defaults):
      gpu, pipeline, steps, batch, latency_per_image_s

    We construct:
      tau(b) = b * latency_per_image_s(b)

    Missing batches are filled by linear interpolation on tau(b).
    Values are clamped to [1, batch_max].

    Returns:
      tau_func(b:int) -> float
    """
    import pandas as pd  # local import to keep base deps light

    df = pd.read_csv(csv_path)

    # Normalize for robust matching
    df[gpu_col] = df[gpu_col].astype(str).map(normalize_gpu_name)
    # pipeline may require model to normalize, but CSV should already have normalized pipeline.
    df[pipeline_col] = df[pipeline_col].astype(str).str.lower().str.strip()

    gpu_n = normalize_gpu_name(gpu)
    pipeline_n = str(pipeline).lower().strip()
    steps = int(steps)
    batch_max = int(batch_max)

    sub = df[(df[gpu_col] == gpu_n) & (df[pipeline_col] == pipeline_n) & (df[steps_col] == steps)].copy()
    if sub.empty:
        # Provide a helpful error showing what exists
        avail = df[[gpu_col, pipeline_col, steps_col]].drop_duplicates().sort_values([gpu_col, pipeline_col, steps_col])
        raise ValueError(
            f"No profile rows for gpu={gpu_n}, pipeline={pipeline_n}, steps={steps} in {csv_path}.\n"
            f"Available (gpu,pipeline,steps) combos (first 20):\n{avail.head(20).to_string(index=False)}"
        )

    # clean and sort
    sub = sub[[batch_col, latency_per_image_col]].dropna()
    sub[batch_col] = sub[batch_col].astype(int)
    sub = sub.sort_values(batch_col)

    # compute tau points
    x = sub[batch_col].to_numpy(dtype=int)
    y = (sub[batch_col].to_numpy(dtype=float) * sub[latency_per_image_col].to_numpy(dtype=float))

    if x.size == 0:
        raise ValueError(f"Profile rows exist but no valid batch/latency points for gpu={gpu_n}, pipeline={pipeline_n}, steps={steps}")

    # Deduplicate batches if any (keep last)
    # (rare, but safe)
    uniq = {}
    for bi, ti in zip(x, y):
        uniq[int(bi)] = float(ti)
    x = np.array(sorted(uniq.keys()), dtype=float)
    y = np.array([uniq[int(bi)] for bi in x], dtype=float)

    def tau_func(b: int) -> float:
        b = int(b)
        b = max(1, min(batch_max, b))
        # exact match
        if float(b) in x:
            # small x array, linear search ok
            return float(y[np.where(x == float(b))[0][0]])
        # interpolate on tau(b)
        return float(np.interp(float(b), x, y, left=float(y[0]), right=float(y[-1])))

    return tau_func


# -------------------- existing utilities --------------------

def default_wait_candidates(W_max=2.0, n=10):
    return list(np.linspace(0.0, float(W_max), int(n)))


def compute_latency_stats(tasks):
    latencies = [t.finish_time - t.arrival_time for t in tasks if t.finish_time is not None]
    latencies = np.asarray(latencies, dtype=float)

    if latencies.size == 0:
        return {
            "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan,
            "p50": np.nan, "p90": np.nan, "p95": np.nan, "p99": np.nan, "count": 0,
        }

    return {
        "mean": float(np.mean(latencies)),
        "std": float(np.std(latencies)),
        "min": float(np.min(latencies)),
        "max": float(np.max(latencies)),
        "p50": float(np.percentile(latencies, 50)),
        "p90": float(np.percentile(latencies, 90)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
        "count": int(latencies.size),
    }


def plot_backlogs(queues, save_path=None, show=True):
    import matplotlib.pyplot as plt

    plt.figure()
    plotted = False

    for idx, q in enumerate(queues):
        if getattr(q, "backlog_log", None) and len(q.backlog_log) > 0:
            t, b = zip(*q.backlog_log)
            plt.plot(t, b, label=f"Queue {idx}")
            plotted = True

    plt.xlabel("Time")
    plt.ylabel("Queue Length")
    plt.legend()
    plt.title("Queue Backlog Over Time")
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return plotted


def summarize_events(logger):
    if logger is None or getattr(logger, "n", 0) == 0:
        return {
            "N_events": 0,
            "avg_batch": np.nan,
            "avg_event_cost": np.nan,
            "avg_cost_per_task_in_batch": np.nan,
            "avg_inter_event": np.nan,
            "arrivals_per_event_mean": np.nan,
            "arrivals_per_event_std": np.nan,
            "total_arrivals_observed": 0,
        }

    n = int(logger.n)
    b = np.asarray(logger.b, dtype=float)
    y = np.asarray(logger.y, dtype=float)
    T = np.asarray(logger.T, dtype=float)

    avg_inter = float(np.mean(np.diff(T))) if n >= 2 else np.nan

    A = np.asarray(logger.A, dtype=float)  # shape (n, K)
    arrivals_per_event = np.sum(A, axis=1)
    total_arrivals = int(np.sum(arrivals_per_event))

    cost_per_task = np.divide(y, b, out=np.full_like(y, np.nan), where=(b > 0))

    return {
        "N_events": n,
        "avg_batch": float(np.mean(b)),
        "avg_event_cost": float(np.mean(y)),
        "avg_cost_per_task_in_batch": float(np.nanmean(cost_per_task)),
        "avg_inter_event": avg_inter,
        "arrivals_per_event_mean": float(np.mean(arrivals_per_event)),
        "arrivals_per_event_std": float(np.std(arrivals_per_event)),
        "total_arrivals_observed": total_arrivals,
    }


def save_results(strategy, stats, cost, out_path="results.csv", logger=None, extra=None):
    import pandas as pd

    row = {"strategy": strategy, "cost_rate": float(cost), **stats}

    if logger is not None:
        row.update(summarize_events(logger))

    if extra is not None:
        for k, v in extra.items():
            row[k] = v

    df_new = pd.DataFrame([row])

    try:
        old = pd.read_csv(out_path)
        df_all = pd.concat([old, df_new], ignore_index=True)
    except FileNotFoundError:
        df_all = df_new

    df_all.to_csv(out_path, index=False)
