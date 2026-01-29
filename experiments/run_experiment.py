# run_experiment.py
import simpy
import argparse
import numpy as np

from simulator.async_dispatch_simulator import (
    GPUQueue,
    TaskGenerator,
    TimestampTaskGenerator,
    Dispatcher,
    GlobalEventLogger,
)

from arc.routing import make_routing_policy
from arc.batching import ThresholdBatching

from scripts.config_and_utils import (
    default_wait_candidates,
    compute_latency_stats,
    plot_backlogs,
    save_results,
    sublinear_tau_fn,
    build_tau_from_profile_csv,
)


# ---------- parsing helpers ----------

def _parse_float_list(s: str):
    if s is None or str(s).strip() == "":
        return []
    return [float(x.strip()) for x in str(s).split(",") if x.strip() != ""]

def _parse_int_list(s: str):
    if s is None or str(s).strip() == "":
        return []
    return [int(x.strip()) for x in str(s).split(",") if x.strip() != ""]

def _parse_str_list(s: str):
    if s is None or str(s).strip() == "":
        return []
    return [x.strip() for x in str(s).split(",") if x.strip() != ""]

def _expand_list_to_K(vals, K):
    vals = list(vals)
    if len(vals) == 0:
        raise ValueError("Empty list given.")
    if len(vals) >= K:
        return vals[:K]
    return vals + [vals[-1]] * (K - len(vals))

def _normalize_arrival_times(arrival_times, normalize: bool = True):
    arr = [float(t) for t in arrival_times]
    if len(arr) == 0:
        return arr
    if not normalize:
        return arr
    t0 = arr[0]
    return [t - t0 for t in arr]

def _load_trace_from_npy(
    trace_npy: str,
    trace_day: int = -1,
    trace_stride: int = 1,
    trace_scale: float = 1.0,
):
    """
    Load arrival_times.npy and optionally:
      - select one day: [day*86400, (day+1)*86400) and rebase to 0 (assumes trace already starts at 0)
      - subsample by stride
      - scale time axis
    Returns: Python list[float]
    """
    arr = np.load(trace_npy)
    arr = np.asarray(arr, dtype=np.float64)

    if arr.size == 0:
        return []

    if int(trace_day) >= 0:
        d = int(trace_day)
        start = d * 86400.0
        end = (d + 1) * 86400.0
        arr = arr[(arr >= start) & (arr < end)] - start

    stride = max(1, int(trace_stride))
    if stride > 1:
        arr = arr[::stride]

    scale = float(trace_scale)
    if scale != 1.0:
        arr = arr * scale

    return arr.tolist()

def _percentile_safe(x: np.ndarray, p: float):
    if x.size == 0:
        return np.nan
    return float(np.percentile(x, p))


def run_sim(
    routing="arc",
    sim_time=1000.0,
    K=2,
    arrival_rate=1.2,
    V_param=10.0,
    save_csv=True,
    seed=0,

    # per-GPU heterogeneity
    B_list=None,
    p_list=None,
    tau_alpha_list=None,
    tau_beta_list=None,
    tau_gamma_list=None,

    # dispatch-time knobs (NOTE: W_max reused by threshold batching)
    W_max=2.0,
    W_n=10,
    arrival_estimation_window=50.0,

    # backlog sampling
    backlog_sample_period=1.0,

    # trace arrivals
    arrival_times=None,
    normalize_trace=True,

    # run-to-end behavior for trace
    trace_run_to_end=True,
    trace_end_cushion=10.0,

    # hetero empirical profiles
    profile_csv="",
    gpu_names_list=None,
    pipeline_list=None,
    steps_list=None,

    # batching baseline switch
    batching="arc",  # {"arc","threshold"}

    # bandit loss weights
    bandit_alpha_cost=0.0,  # loss = latency + alpha * cost_per_task + beta * Q_route_time
    bandit_beta_queue=0.0,  # (C2) beta * Q_route_time

    # ===== tunable bandit hyperparams =====
    bandit_ucb_c=1.0,
    linucb_alpha=1.0,
    linucb_ridge=1.0,
    linucb_bias=True,
):
    np.random.seed(int(seed))
    K = int(K)

    # ----- defaults -----
    if B_list is None:
        B_list = [8]
    if p_list is None:
        p_list = [1.0, 1.8]
    if tau_alpha_list is None:
        tau_alpha_list = [0.2]
    if tau_beta_list is None:
        tau_beta_list = [0.1]
    if tau_gamma_list is None:
        tau_gamma_list = [0.7]

    if gpu_names_list is None:
        gpu_names_list = []
    if pipeline_list is None:
        pipeline_list = []
    if steps_list is None:
        steps_list = []

    # expand to length K
    B_list = _expand_list_to_K(B_list, K)
    p_list = _expand_list_to_K(p_list, K)
    tau_alpha_list = _expand_list_to_K(tau_alpha_list, K)
    tau_beta_list = _expand_list_to_K(tau_beta_list, K)
    tau_gamma_list = _expand_list_to_K(tau_gamma_list, K)

    WAIT_CANDIDATES = default_wait_candidates(W_max=W_max, n=W_n)

    env = simpy.Environment()
    logger = GlobalEventLogger(env, K=K)

    # --- routing policy (tunable factory) ---
    routing_policy = make_routing_policy(
        name=routing,
        V=float(V_param),
        bandit_ucb_c=float(bandit_ucb_c),
        linucb_alpha=float(linucb_alpha),
        linucb_ridge=float(linucb_ridge),
        linucb_bias=bool(linucb_bias),
    )

    # --- batching policy ---
    batching = str(batching).lower().strip()
    if batching not in ["arc", "threshold"]:
        raise ValueError("batching must be one of: arc, threshold")

    # Preserve original behavior when batching="arc" by passing None
    if batching == "arc":
        batching_policy = None
    else:
        batching_policy = ThresholdBatching(Wmax=float(W_max))

    # build per-GPU tau_k
    tau_funcs = []
    use_profile = (str(profile_csv).strip() != "")

    if use_profile:
        if len(gpu_names_list) == 0 or len(pipeline_list) == 0 or len(steps_list) == 0:
            raise ValueError(
                "When using --profile_csv, you must provide --gpu_names_list, --pipeline_list, --steps_list "
                "(comma-separated)."
            )

        gpu_names_list = _expand_list_to_K(gpu_names_list, K)
        pipeline_list = _expand_list_to_K(pipeline_list, K)
        steps_list = _expand_list_to_K(steps_list, K)

        for k in range(K):
            tau_k = build_tau_from_profile_csv(
                csv_path=str(profile_csv).strip(),
                gpu=str(gpu_names_list[k]),
                pipeline=str(pipeline_list[k]),
                steps=int(steps_list[k]),
                batch_max=int(B_list[k]),
            )
            tau_funcs.append(tau_k)
    else:
        for k in range(K):
            a = float(tau_alpha_list[k])
            b0 = float(tau_beta_list[k])
            g = float(tau_gamma_list[k])
            tau_funcs.append(lambda bb, a=a, b0=b0, g=g: sublinear_tau_fn(bb, alpha=a, beta=b0, gamma=g))

    # build queues (pass batching_policy + routing_policy + bandit weights)
    queues = []
    for k in range(K):
        queues.append(
            GPUQueue(
                env=env,
                idx=k,
                tau_func=tau_funcs[k],
                B_k=int(B_list[k]),
                p_k=float(p_list[k]),
                wait_candidates=WAIT_CANDIDATES,
                V=float(V_param),
                logger=logger,
                arrival_estimation_window=float(arrival_estimation_window),
                backlog_sample_period=float(backlog_sample_period),

                # NEW
                batching_policy=batching_policy,
                routing_policy=routing_policy,
                bandit_alpha_cost=float(bandit_alpha_cost),
                bandit_beta_queue=float(bandit_beta_queue),
            )
        )

    dispatcher = Dispatcher(
        env=env,
        gpu_queues=queues,
        logger=logger,
        routing_policy=routing_policy,
    )

    # workload generator
    trace_mode = (arrival_times is not None)

    if trace_mode:
        trace = _normalize_arrival_times(arrival_times, normalize=normalize_trace)

        if len(trace) > 0:
            if trace_run_to_end:
                duration = float(trace[-1]) + float(trace_end_cushion)
            else:
                duration = float(sim_time)

            generator = TimestampTaskGenerator(
                env=env,
                dispatcher=dispatcher,
                arrival_times=trace,
                duration=float(duration),
            )
        else:
            duration = float(sim_time)
            generator = TimestampTaskGenerator(
                env=env,
                dispatcher=dispatcher,
                arrival_times=[],
                duration=float(duration),
            )
    else:
        duration = float(sim_time)
        generator = TaskGenerator(
            env=env,
            dispatcher=dispatcher,
            arrival_rate=float(arrival_rate),
            duration=float(duration),
        )

    # run sim
    env.run(until=float(duration))

    # ---------------- metrics ----------------
    stats = compute_latency_stats(generator.tasks)
    total_time = float(duration)
    cost = sum(q.busy_time * q.p_k for q in queues) / total_time if total_time > 0 else np.nan

    # extra diagnostics
    backlog_summaries = []
    for q in queues:
        if getattr(q, "backlog_log", None) and len(q.backlog_log) > 0:
            qs = np.array([b for (_, b) in q.backlog_log], dtype=float)
            backlog_summaries.append({
                "mean": float(qs.mean()),
                "p90": _percentile_safe(qs, 90),
                "p95": _percentile_safe(qs, 95),
                "p99": _percentile_safe(qs, 99),
                "max": float(qs.max()),
            })
        else:
            backlog_summaries.append({"mean": np.nan, "p90": np.nan, "p95": np.nan, "p99": np.nan, "max": np.nan})

    batch_summaries = []
    for q in queues:
        total_batches = int(getattr(q, "num_batches", 0))
        total_done = int(getattr(q, "num_completed", 0))
        avg_batch = (total_done / total_batches) if total_batches > 0 else np.nan
        eff_throughput = (total_done / total_time) if total_time > 0 else np.nan
        util = (q.busy_time / total_time) if total_time > 0 else np.nan
        batch_summaries.append({
            "num_batches": total_batches,
            "num_completed": total_done,
            "avg_batch": float(avg_batch) if not np.isnan(avg_batch) else np.nan,
            "eff_throughput_tasks_per_s": float(eff_throughput) if not np.isnan(eff_throughput) else np.nan,
            "util": float(util) if not np.isnan(util) else np.nan,
        })

    # ARC wait-time stats only meaningful when routing == "arc" and batching == "arc"
    w_stats = []
    if (routing == "arc") and (batching == "arc"):
        for q in queues:
            ws = []
            for w_val, cnt in q.w_hist.items():
                ws.extend([float(w_val)] * int(cnt))
            ws = np.asarray(ws, dtype=float)
            w_stats.append({
                "w_mean": float(ws.mean()) if ws.size else np.nan,
                "w_p50": _percentile_safe(ws, 50),
                "w_p90": _percentile_safe(ws, 90),
                "w_p99": _percentile_safe(ws, 99),
                "w_count": int(ws.size),
                "top_w": q.w_hist.most_common(5),
            })
    else:
        w_stats = [None for _ in queues]

    print(f"Routing: {routing}")
    print(f"Batching: {batching}")
    print(f"Simulated time horizon: {total_time:.3f}")
    print(f"Latency stats: {stats}")
    print(f"Busy cost rate: {cost:.6f}")

    # show bandit config if used
    if routing in ["mab_ucb1", "cb_linucb"]:
        print(
            "Bandit loss: latency + alpha*cost_per_task + beta*Q_route_time "
            f"(reward=-loss). alpha={float(bandit_alpha_cost):.6g}, beta={float(bandit_beta_queue):.6g}"
        )
        if routing == "mab_ucb1":
            print(f"  UCB1 hyperparam: c={float(bandit_ucb_c):.6g}")
        if routing == "cb_linucb":
            print(
                f"  LinUCB hyperparams: ucb_alpha={float(linucb_alpha):.6g}, "
                f"ridge={float(linucb_ridge):.6g}, bias={bool(linucb_bias)}"
            )

    print("\n[Per-GPU configs]:")
    for k, q in enumerate(queues):
        if use_profile:
            print(
                f"GPU {k}: gpu={gpu_names_list[k]}, pipeline={pipeline_list[k]}, steps={steps_list[k]}, "
                f"B_k={q.B_k}, p_k={q.p_k}"
            )
        else:
            print(f"GPU {k}: B_k={q.B_k}, p_k={q.p_k}")

    print("\n[Per-GPU utilization / throughput]:")
    for k, summ in enumerate(batch_summaries):
        print(
            f"GPU {k}: util={summ['util']:.4f}, eff_throughput={summ['eff_throughput_tasks_per_s']:.4f} task/s, "
            f"avg_batch={summ['avg_batch']:.3f}, batches={summ['num_batches']}, completed={summ['num_completed']}"
        )

    print("\n[Per-GPU backlog summary] (sampled):")
    for k, summ in enumerate(backlog_summaries):
        print(
            f"GPU {k}: mean={summ['mean']:.2f}, p90={summ['p90']:.2f}, p95={summ['p95']:.2f}, "
            f"p99={summ['p99']:.2f}, max={summ['max']:.0f}"
        )

    print("\n[Per-GPU batch-size histogram] (top 8):")
    for k, q in enumerate(queues):
        top = q.batch_hist.most_common(8) if getattr(q, "batch_hist", None) is not None else []
        print(f"GPU {k}: {top}")

    print("\n[Routing histogram]:")
    for gpu_id, count in sorted(logger.routing_decisions.items()):
        print(f"GPU {gpu_id}: {count} tasks")

    if (routing == "arc") and (batching == "arc"):
        print("\n[ARC wait-time (w) stats per GPU] (routing=arc, batching=arc):")
        for k, st in enumerate(w_stats):
            if st is None:
                continue
            top_w_str = ", ".join([f"{w}:{c}" for (w, c) in st["top_w"]])
            print(
                f"GPU {k}: w_mean={st['w_mean']:.6g}, p50={st['w_p50']:.6g}, p90={st['w_p90']:.6g}, "
                f"p99={st['w_p99']:.6g}, count={st['w_count']} | top_w={top_w_str}"
            )

    if save_csv:
        extra = {
            "V": float(V_param),
            "K": int(K),
            "arrival_rate": float(arrival_rate) if not trace_mode else np.nan,
            "seed": int(seed),
            "B_list": ",".join(str(int(x)) for x in B_list),
            "p_list": ",".join(str(float(x)) for x in p_list),
            "tau_alpha_list": ",".join(str(float(x)) for x in tau_alpha_list),
            "tau_beta_list": ",".join(str(float(x)) for x in tau_beta_list),
            "tau_gamma_list": ",".join(str(float(x)) for x in tau_gamma_list),
            "W_max": float(W_max),
            "W_n": int(W_n),
            "arrival_estimation_window": float(arrival_estimation_window),
            "backlog_sample_period": float(backlog_sample_period),
            "trace_mode": int(trace_mode),
            "sim_horizon": float(total_time),

            "use_profile": int(use_profile),
            "profile_csv": str(profile_csv).strip() if use_profile else "",

            # NEW
            "batching": str(batching),
            "bandit_alpha_cost": float(bandit_alpha_cost),
            "bandit_beta_queue": float(bandit_beta_queue),

            # bandit hyperparams
            "bandit_ucb_c": float(bandit_ucb_c),
            "linucb_alpha": float(linucb_alpha),
            "linucb_ridge": float(linucb_ridge),
            "linucb_bias": int(bool(linucb_bias)),
        }
        if use_profile:
            extra.update({
                "gpu_names_list": ",".join(gpu_names_list),
                "pipeline_list": ",".join(pipeline_list),
                "steps_list": ",".join(str(int(x)) for x in steps_list),
            })

        for k, summ in enumerate(batch_summaries):
            extra[f"gpu{k}_util"] = summ["util"]
            extra[f"gpu{k}_eff_tps"] = summ["eff_throughput_tasks_per_s"]
            extra[f"gpu{k}_avg_batch"] = summ["avg_batch"]
            extra[f"gpu{k}_num_batches"] = summ["num_batches"]
            extra[f"gpu{k}_num_completed"] = summ["num_completed"]

        for k, summ in enumerate(backlog_summaries):
            extra[f"gpu{k}_Qmean"] = summ["mean"]
            extra[f"gpu{k}_Qp95"] = summ["p95"]
            extra[f"gpu{k}_Qp99"] = summ["p99"]
            extra[f"gpu{k}_Qmax"] = summ["max"]

        if (routing == "arc") and (batching == "arc"):
            for k, st in enumerate(w_stats):
                if st is None:
                    continue
                extra[f"gpu{k}_w_mean"] = st["w_mean"]
                extra[f"gpu{k}_w_p90"] = st["w_p90"]
                extra[f"gpu{k}_w_p99"] = st["w_p99"]
                extra[f"gpu{k}_w_count"] = st["w_count"]

        save_results(strategy=routing, stats=stats, cost=cost, logger=logger, extra=extra)

    plot_backlogs(queues)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # routing (UPDATED: include jsw)
    parser.add_argument(
        "--routing",
        type=str,
        default="arc",
        choices=["arc", "jsq", "jsw", "rr", "random", "mab_ucb1", "cb_linucb"],
    )

    # batching
    parser.add_argument(
        "--batching",
        type=str,
        default="arc",
        choices=["arc", "threshold"],
        help="arc: original ARC wait-time; threshold: run when batch full (B_k) or wait reaches W_max.",
    )

    # bandit loss weights
    parser.add_argument(
        "--bandit_alpha_cost",
        type=float,
        default=0.0,
        help="Alpha in loss = latency + alpha * cost_per_task + beta * Q_route_time (reward=-loss).",
    )
    parser.add_argument(
        "--bandit_beta_queue",
        type=float,
        default=0.0,
        help="Beta in loss term beta * Q_route_time (C2).",
    )

    # ===== bandit hyperparams =====
    parser.add_argument("--bandit_ucb_c", type=float, default=1.0, help="UCB1 exploration coefficient c.")
    parser.add_argument("--linucb_alpha", type=float, default=1.0, help="LinUCB exploration coefficient.")
    parser.add_argument("--linucb_ridge", type=float, default=1.0, help="LinUCB ridge regularization lambda.")
    parser.add_argument(
        "--linucb_bias",
        action="store_true",
        help="If set, include a bias term in LinUCB context. (Default: False unless you pass this flag.)",
    )
    parser.add_argument(
        "--linucb_no_bias",
        action="store_true",
        help="If set, explicitly disable bias term. Overrides --linucb_bias.",
    )

    # sim setup
    parser.add_argument("--time", type=float, default=1000.0)
    parser.add_argument("--V", type=float, default=10.0)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--K", type=int, default=2)

    # Poisson arrivals
    parser.add_argument("--arrival_rate", type=float, default=1.2)

    # per-GPU lists
    parser.add_argument("--B_list", type=str, default="8")
    parser.add_argument("--p_list", type=str, default="1.0,1.8")
    parser.add_argument("--tau_alpha_list", type=str, default="0.2")
    parser.add_argument("--tau_beta_list", type=str, default="0.1")
    parser.add_argument("--tau_gamma_list", type=str, default="0.7")

    # dispatch-time knobs (W_max reused by threshold batching)
    parser.add_argument("--Wmax", type=float, default=2.0)
    parser.add_argument("--Wn", type=int, default=10)
    parser.add_argument("--arrival_window", type=float, default=50.0)
    parser.add_argument("--backlog_sample_period", type=float, default=1.0)

    # quick inline trace
    parser.add_argument("--arrival_times", type=str, default="")
    parser.add_argument("--normalize_trace", action="store_true")

    # real trace from .npy
    parser.add_argument("--trace_npy", type=str, default="", help="Path to arrival_times.npy")
    parser.add_argument("--trace_day", type=int, default=-1)
    parser.add_argument("--trace_stride", type=int, default=1)
    parser.add_argument("--trace_scale", type=float, default=1.0)

    # trace horizon control
    parser.add_argument("--trace_run_to_end", action="store_true")
    parser.add_argument("--trace_cushion", type=float, default=10.0)

    # hetero profiles
    parser.add_argument("--profile_csv", type=str, default="",
                        help="CSV exported from plot script. If set, overrides sublinear tau.")
    parser.add_argument("--gpu_names_list", type=str, default="",
                        help="Comma-separated: A100,H100,L40S ... length K (or shorter, will repeat last).")
    parser.add_argument("--pipeline_list", type=str, default="",
                        help="Comma-separated: sdxl,sdxl,sd15 ... length K (or shorter, will repeat last).")
    parser.add_argument("--steps_list", type=str, default="",
                        help="Comma-separated ints: 20,20,20 ... length K (or shorter, will repeat last).")

    args = parser.parse_args()

    B_list = _parse_int_list(args.B_list)
    p_list = _parse_float_list(args.p_list)
    a_list = _parse_float_list(args.tau_alpha_list)
    b_list = _parse_float_list(args.tau_beta_list)
    g_list = _parse_float_list(args.tau_gamma_list)

    # arrivals: trace_npy > inline
    arrival_times = None
    if args.trace_npy.strip() != "":
        arrival_times = _load_trace_from_npy(
            trace_npy=args.trace_npy.strip(),
            trace_day=int(args.trace_day),
            trace_stride=int(args.trace_stride),
            trace_scale=float(args.trace_scale),
        )
    elif args.arrival_times.strip() != "":
        arrival_times = _parse_float_list(args.arrival_times)

    gpu_names_list = _parse_str_list(args.gpu_names_list)
    pipeline_list = _parse_str_list(args.pipeline_list)
    steps_list = _parse_int_list(args.steps_list)

    # resolve linucb bias flags
    linucb_bias = bool(args.linucb_bias)
    if bool(args.linucb_no_bias):
        linucb_bias = False

    run_sim(
        routing=args.routing,
        batching=args.batching,

        bandit_alpha_cost=float(args.bandit_alpha_cost),
        bandit_beta_queue=float(args.bandit_beta_queue),

        bandit_ucb_c=float(args.bandit_ucb_c),
        linucb_alpha=float(args.linucb_alpha),
        linucb_ridge=float(args.linucb_ridge),
        linucb_bias=bool(linucb_bias),

        sim_time=float(args.time),
        K=int(args.K),
        arrival_rate=float(args.arrival_rate),
        V_param=float(args.V),
        save_csv=bool(args.save),
        seed=int(args.seed),

        B_list=B_list,
        p_list=p_list,
        tau_alpha_list=a_list,
        tau_beta_list=b_list,
        tau_gamma_list=g_list,

        W_max=float(args.Wmax),
        W_n=int(args.Wn),
        arrival_estimation_window=float(args.arrival_window),
        backlog_sample_period=float(args.backlog_sample_period),

        arrival_times=arrival_times,
        normalize_trace=bool(args.normalize_trace),

        trace_run_to_end=bool(args.trace_run_to_end),
        trace_end_cushion=float(args.trace_cushion),

        profile_csv=args.profile_csv,
        gpu_names_list=gpu_names_list,
        pipeline_list=pipeline_list,
        steps_list=steps_list,
    )

