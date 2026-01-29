# async_dispatch_simulator.py
import simpy
import numpy as np
from collections import deque, Counter
from typing import Callable, List, Dict, Optional, Any


class Task:
    def __init__(self, arrival_time: float):
        self.arrival_time = float(arrival_time)
        self.start_time = None
        self.finish_time = None

        # NEW (for bandit baselines; does not affect existing logic)
        self.routed_gpu: Optional[int] = None
        self.bandit_context: Optional[np.ndarray] = None  # context vector used at routing time
        self.route_time_Q: Optional[float] = None  # NEW (C2): queue length at routing time


class GlobalEventLogger:
    """
    Records routing histogram and (optional) per-completion event summaries.
    """
    def __init__(self, env: simpy.Environment, K: int):
        self.env = env
        self.K = int(K)
        self.n = 0
        self.T = []
        self.I = []
        self.b = []
        self.y = []
        self.A = []
        self._arrivals_since_last = np.zeros(self.K, dtype=int)
        self.routing_decisions = Counter()

    def record_arrival(self, k: int):
        kk = int(k)
        self._arrivals_since_last[kk] += 1
        self.routing_decisions[kk] += 1

    def record_completion(self, i: int, b_n: int, y_n: float):
        A_vec = self._arrivals_since_last.copy()
        self.T.append(float(self.env.now))
        self.I.append(int(i))
        self.b.append(int(b_n))
        self.y.append(float(y_n))
        self.A.append(A_vec)
        self._arrivals_since_last[:] = 0
        self.n += 1


# -------------------- Batching policies (NEW; preserves original behavior) --------------------

class BatchingPolicy:
    """
    Hook that decides how long to wait before starting the next batch when a GPU becomes idle.
    """
    name = "base"

    def wait_before_service(self, q: "GPUQueue") -> float:
        # Default: no wait
        return 0.0

    def wait_before_service_event(self, q: "GPUQueue"):
        """
        Optional event-driven waiting (yieldable) for policies like threshold batching.
        If implemented, GPUQueue.run() will use it instead of wait_before_service().
        """
        raise NotImplementedError


class ARCBatching(BatchingPolicy):
    """
    Uses the existing ARC wait-time chooser in GPUQueue.choose_wait_time_arc().
    """
    name = "arc"

    def wait_before_service(self, q: "GPUQueue") -> float:
        return float(q.choose_wait_time_arc())


class ThresholdBatching(BatchingPolicy):
    """
    Threshold batching: when GPU becomes idle and queue non-empty, wait until:
      - batch is full (len(queue) >= B_k), OR
      - waiting reaches Wmax
    then start service immediately.
    """
    name = "threshold"

    def __init__(self, Wmax: float):
        self.Wmax = float(Wmax)

    def wait_before_service_event(self, q: "GPUQueue"):
        if len(q.queue) <= 0:
            return
        if len(q.queue) >= q.B_k:
            return

        start = float(q.env.now)
        while True:
            if len(q.queue) >= q.B_k:
                return

            elapsed = float(q.env.now) - start
            remain = self.Wmax - elapsed
            if remain <= 0:
                return

            # Wait for either a new arrival to this queue, or timeout
            res = yield q.env.any_of([q.env.timeout(remain), q._arrival_event])
            # If timeout happened, we're done; else loop to see if full batch achieved
            if any(isinstance(ev, simpy.events.Timeout) for ev in res.keys()):
                return


class GPUQueue:
    r"""
    One GPU worker with:
      - its own batching cap B_k
      - its own per-unit busy price p_k
      - its own latency curve tau_k(b)
    and ARC dispatch-time waiting control.

    Dispatch-time rule (paper):
      hat_lambda_k(t): empirical arrival rate into queue k over sliding window
      for each candidate w:
        hat_b = min{ Q + hat_lambda*w, B_k }
        hat_tau = tau_k(hat_b)
        hat_a = hat_lambda*(w + hat_tau)
        Psi = ( (Q/hat_mu)*(hat_b - hat_a) - V*p_k*hat_tau ) / (w + hat_tau)
      choose w maximizing Psi

    Notes:
      - arrival_estimation_window <= 0 => lambda hat is 0 (ARC waiting degenerates to w=0 typically).
      - backlog_log: now sampled periodically (not recorded per completion) to avoid huge logs for long traces.
    """
    def __init__(
        self,
        env: simpy.Environment,
        idx: int,
        tau_func: Callable[[int], float],
        B_k: int,
        p_k: float,
        wait_candidates: List[float],
        V: float,
        logger: GlobalEventLogger,
        arrival_estimation_window: float = 50.0,
        idle_poll: float = 0.1,
        backlog_sample_period: float = 1.0,

        # NEW (optional; preserves original defaults)
        batching_policy: Optional[BatchingPolicy] = None,
        routing_policy: Optional["RoutingPolicy"] = None,  # for bandit update hooks
        bandit_alpha_cost: float = 0.0,  # alpha in (latency + alpha * cost_per_task)
        bandit_beta_queue: float = 0.0,  # beta in (.. + beta * Q_route_time)
        recent_latency_ewma_beta: float = 0.9,  # EWMA smoothing for "recent latency" context feature
    ):
        self.env = env
        self.idx = int(idx)

        self.queue = deque()
        self.tau = tau_func
        self.B_k = int(B_k)
        self.p_k = float(p_k)
        self.W = [float(w) for w in wait_candidates]
        self.V = float(V)

        self.logger = logger

        # For hat_lambda_k(t): record timestamps of arrivals *to this queue*.
        self.arrival_estimation_window = float(arrival_estimation_window)
        self._arrival_times = deque()

        self.idle_poll = float(idle_poll)

        # Stats/logs
        self.busy_time = 0.0

        # periodic backlog sampling
        self.backlog_log = []
        self.backlog_sample_period = float(backlog_sample_period)

        # batch/service stats
        self.num_completed = 0
        self.num_batches = 0
        self.batch_hist = Counter()

        # Debug / analysis: histogram for chosen w* and observed lambda/mu
        self.w_hist = Counter()
        self.lam_hist = []
        self.mu_hist = []

        # arrival event used by event-driven batching policies (threshold)
        self._arrival_event = env.event()

        # policy hooks
        self.batching_policy = batching_policy  # if None => use ARC wait-time chooser (original)
        self.routing_policy = routing_policy
        self.bandit_alpha_cost = float(bandit_alpha_cost)
        self.bandit_beta_queue = float(bandit_beta_queue)

        # recent latency summary for contextual bandit feature
        self.recent_latency_ewma_beta = float(recent_latency_ewma_beta)
        self.recent_latency_ewma = 0.0
        self.recent_latency_count = 0

        # processes
        self._sampler_action = env.process(self._sample_backlog())
        self.action = env.process(self.run())

    def _sample_backlog(self):
        while True:
            self.backlog_log.append((float(self.env.now), int(len(self.queue))))
            yield self.env.timeout(self.backlog_sample_period)

    def enqueue(self, task: Task):
        self.queue.append(task)
        self._arrival_times.append(float(task.arrival_time))

        # wake any threshold-batching wait
        if not self._arrival_event.triggered:
            self._arrival_event.succeed()
        self._arrival_event = self.env.event()

    # ---------- ARC helper functions ----------

    def _purge_old_arrivals(self, now: float):
        if self.arrival_estimation_window <= 0:
            return
        cutoff = now - self.arrival_estimation_window
        while self._arrival_times and self._arrival_times[0] < cutoff:
            self._arrival_times.popleft()

    def estimate_lambda(self, now: float) -> float:
        """
        hat_lambda_k(t): empirical avg arrival rate into this queue over sliding window.
        """
        T = self.arrival_estimation_window
        if T <= 0:
            return 0.0
        self._purge_old_arrivals(now)
        return float(len(self._arrival_times)) / float(T)

    def tilde_B(self, Q: int) -> int:
        """
        \tilde B_k(t) = min{B_k, Q_k(t)+1}
        """
        return int(min(self.B_k, int(Q) + 1))

    def hat_mu(self, Q: int) -> float:
        """
        \hat\mu_k(t) = max_{1<=b<=\tilde B_k(t)} b/tau_k(b)
        """
        tb = self.tilde_B(Q)
        if tb <= 0:
            return 1e-9

        best = 1e-9
        for b in range(1, tb + 1):
            t = float(self.tau(b))
            if t > 0:
                best = max(best, float(b) / t)
        return best

    def rho(self, Q: int) -> float:
        """
        rho_k(t) = p_k * tau_k(tilde_B_k(t)) / tilde_B_k(t)
        """
        tb = self.tilde_B(Q)
        if tb <= 0:
            return 0.0
        return self.p_k * float(self.tau(tb)) / float(tb)

    def tau_hat(self, b_hat: float) -> float:
        """
        Theory uses tau_k(hat_b) for possibly non-integer hat_b.
        Simulation tau_k(.) defined on integers, so we interpolate linearly.
        """
        if self.B_k <= 0:
            return 0.0

        if b_hat <= 1.0:
            return float(self.tau(1))
        if b_hat >= float(self.B_k):
            return float(self.tau(self.B_k))

        lo = int(np.floor(b_hat))
        hi = int(np.ceil(b_hat))
        lo = max(1, min(lo, self.B_k))
        hi = max(1, min(hi, self.B_k))
        if lo == hi:
            return float(self.tau(lo))

        t_lo = float(self.tau(lo))
        t_hi = float(self.tau(hi))
        w = float(b_hat - lo) / float(hi - lo)
        return (1.0 - w) * t_lo + w * t_hi

    def choose_wait_time_arc(self) -> float:
        """
        Implements w* in argmax_{w in W} Psi_k(w;t).
        Also records chosen w* and lambda/mu for debugging.
        """
        Q = int(len(self.queue))
        if Q <= 0:
            return 0.0

        now = float(self.env.now)
        lam = self.estimate_lambda(now)
        mu = self.hat_mu(Q)

        best_score = -np.inf
        best_w = 0.0

        for w in self.W:
            w = float(w)

            # hat_b = min{ Q + lam*w, B_k }
            b_hat = min(float(Q) + lam * w, float(self.B_k))

            # hat_tau = tau_k(hat_b)
            tau_h = self.tau_hat(b_hat)

            # hat_a = lam*(w + hat_tau)
            a_hat = lam * (w + tau_h)

            denom = (w + tau_h)
            if denom <= 0:
                continue

            numer = (float(Q) / mu) * (b_hat - a_hat) - self.V * self.p_k * tau_h
            score = numer / denom

            if score > best_score:
                best_score = score
                best_w = w

        # debug record
        self.w_hist[float(best_w)] += 1
        self.lam_hist.append(float(lam))
        self.mu_hist.append(float(mu))

        return float(best_w)

    # ---------- Sim loop ----------

    def _update_recent_latency(self, latency: float):
        # EWMA for contextual bandit feature
        lat = float(latency)
        if self.recent_latency_count == 0:
            self.recent_latency_ewma = lat
        else:
            beta = self.recent_latency_ewma_beta
            self.recent_latency_ewma = beta * self.recent_latency_ewma + (1.0 - beta) * lat
        self.recent_latency_count += 1

    def run(self):
        while True:
            if len(self.queue) == 0:
                yield self.env.timeout(self.idle_poll)
                continue

            # batching policy hook; default preserves original ARC batching behavior
            bp = self.batching_policy
            if bp is None:
                # original behavior: ARC dispatch-time waiting
                w = self.choose_wait_time_arc()
                if w > 0:
                    yield self.env.timeout(w)
            else:
                # event-driven threshold batching OR fixed-w batching
                if hasattr(bp, "wait_before_service_event"):
                    try:
                        yield from bp.wait_before_service_event(self)
                    except NotImplementedError:
                        w = float(bp.wait_before_service(self))
                        if w > 0:
                            yield self.env.timeout(w)
                else:
                    w = float(bp.wait_before_service(self))
                    if w > 0:
                        yield self.env.timeout(w)

            batch_size = min(len(self.queue), self.B_k)

            # mark start times
            for i in range(batch_size):
                self.queue[i].start_time = float(self.env.now)

            busy_duration = float(self.tau(batch_size))
            yield self.env.timeout(busy_duration)

            self.busy_time += busy_duration
            self.num_batches += 1
            self.batch_hist[int(batch_size)] += 1

            # pop tasks
            finished_tasks = []
            for _ in range(batch_size):
                task = self.queue.popleft()
                task.finish_time = float(self.env.now)
                finished_tasks.append(task)

            self.num_completed += int(batch_size)

            # logging cost per completion-event (existing)
            y_n = self.p_k * busy_duration
            self.logger.record_completion(i=self.idx, b_n=batch_size, y_n=y_n)

            # keep recent latency (for contextual bandit context feature)
            for task in finished_tasks:
                latency = float(task.finish_time - task.arrival_time)
                self._update_recent_latency(latency)

            # optional routing policy update hook (for bandit baselines)
            # loss = latency + alpha * cost_per_task + beta * Q_route_time, reward = -loss
            if self.routing_policy is not None and hasattr(self.routing_policy, "update"):
                cost_per_task = (self.p_k * busy_duration) / float(batch_size) if batch_size > 0 else 0.0
                alpha = float(self.bandit_alpha_cost)
                beta = float(self.bandit_beta_queue)
                for task in finished_tasks:
                    latency = float(task.finish_time - task.arrival_time)
                    q_term = float(task.route_time_Q) if task.route_time_Q is not None else 0.0
                    loss = latency + alpha * cost_per_task + beta * q_term
                    self.routing_policy.update(task=task, loss=float(loss))


# -------------------- Routing policies (clean plug-ins) --------------------

class RoutingPolicy:
    name = "base"

    def select_queue(self, queues: List[GPUQueue]) -> GPUQueue:
        raise NotImplementedError

    # Optional for contextual bandits: compute & stash context on the Task in Dispatcher.route
    def compute_context(self, q: GPUQueue) -> Optional[np.ndarray]:
        return None

    # Optional for bandits: update after completion, fed by GPUQueue.run()
    def update(self, task: Task, loss: float):
        # default: no-op
        return


class ARCRouting(RoutingPolicy):
    """
    k* in argmin_k { Q_k(t)/hat_mu_k(t) + V*rho_k(t) }
    """
    name = "arc"

    def __init__(self, V: float):
        self.V = float(V)

    def select_queue(self, queues: List[GPUQueue]) -> GPUQueue:
        scores = []
        for q in queues:
            Q = int(len(q.queue))
            mu = q.hat_mu(Q)
            rho = q.rho(Q)
            scores.append(float(Q) / mu + self.V * rho)
        return queues[int(np.argmin(scores))]


class RandomRouting(RoutingPolicy):
    name = "random"

    def select_queue(self, queues: List[GPUQueue]) -> GPUQueue:
        return np.random.choice(queues)


class RoundRobinRouting(RoutingPolicy):
    name = "rr"

    def __init__(self):
        self._i = 0

    def select_queue(self, queues: List[GPUQueue]) -> GPUQueue:
        q = queues[self._i % len(queues)]
        self._i += 1
        return q


class JSQRouting(RoutingPolicy):
    name = "jsq"

    def select_queue(self, queues: List[GPUQueue]) -> GPUQueue:
        return min(queues, key=lambda qq: len(qq.queue))


class JSWRouting(RoutingPolicy):
    """
    JSW / JSQ-μ: choose k minimizing Q_k(t) / hat_mu_k(t).
    """
    name = "jsw"

    def select_queue(self, queues: List[GPUQueue]) -> GPUQueue:
        scores = []
        for q in queues:
            Q = int(len(q.queue))
            mu = float(q.hat_mu(Q))
            mu = max(mu, 1e-9)
            scores.append(float(Q) / mu)
        return queues[int(np.argmin(scores))]


# -------------------- Bandit baselines (tunable hyperparams) --------------------

class UCB1BanditRouting(RoutingPolicy):
    """
    Naive non-contextual MAB with UCB1.
    Maximizes reward r = -loss, where loss is fed via update().
    """
    name = "mab_ucb1"

    def __init__(self, c: float = 1.0):
        self.c = float(c)
        self.counts: Optional[np.ndarray] = None
        self.means: Optional[np.ndarray] = None
        self.t = 0

    def _init_if_needed(self, K: int):
        if self.counts is None or self.means is None:
            self.counts = np.zeros(K, dtype=int)
            self.means = np.zeros(K, dtype=float)
            self.t = 0

    def select_queue(self, queues: List[GPUQueue]) -> GPUQueue:
        K = len(queues)
        self._init_if_needed(K)
        self.t += 1

        for k in range(K):
            if self.counts[k] == 0:
                return queues[k]

        ucb = self.means + self.c * np.sqrt(np.log(self.t) / self.counts)
        return queues[int(np.argmax(ucb))]

    def update(self, task: Task, loss: float):
        if task.routed_gpu is None:
            return
        k = int(task.routed_gpu)
        if self.counts is None or self.means is None:
            return
        r = -float(loss)
        n = int(self.counts[k])
        self.counts[k] = n + 1
        self.means[k] = (self.means[k] * n + r) / float(n + 1)


class LinUCBContextualRouting(RoutingPolicy):
    """
    Contextual bandit baseline using per-arm LinUCB.

    context = [Q_k, util_k, recent_latency_k] (+ optional bias)
    Maximizes reward r = -loss (fed via update()).
    """
    name = "cb_linucb"

    def __init__(
        self,
        ucb_alpha: float = 1.0,
        ridge: float = 1.0,
        include_bias: bool = True,
    ):
        self.ucb_alpha = float(ucb_alpha)
        self.ridge = float(ridge)
        self.include_bias = bool(include_bias)

        self.A: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        self.d: Optional[int] = None

    def _context_vec(self, q: GPUQueue) -> np.ndarray:
        Q = float(len(q.queue))
        now = float(q.env.now)
        util = float(q.busy_time / now) if now > 1e-12 else 0.0
        recent_lat = float(getattr(q, "recent_latency_ewma", 0.0))

        x = np.array([Q, util, recent_lat], dtype=float)
        if self.include_bias:
            x = np.concatenate([np.ones(1, dtype=float), x])
        return x

    def _init_if_needed(self, K: int, d: int):
        if self.d is None:
            self.d = int(d)
        if len(self.A) != K:
            self.A = [self.ridge * np.eye(self.d, dtype=float) for _ in range(K)]
            self.b = [np.zeros(self.d, dtype=float) for _ in range(K)]

    def compute_context(self, q: GPUQueue) -> Optional[np.ndarray]:
        return self._context_vec(q)

    def select_queue(self, queues: List[GPUQueue]) -> GPUQueue:
        K = len(queues)
        x0 = self._context_vec(queues[0])
        self._init_if_needed(K, d=int(x0.size))

        scores = []
        for k, q in enumerate(queues):
            x = self._context_vec(q)
            A_inv = np.linalg.inv(self.A[k])
            theta = A_inv @ self.b[k]
            mean = float(theta @ x)
            bonus = float(self.ucb_alpha * np.sqrt(x @ A_inv @ x))
            scores.append(mean + bonus)

        return queues[int(np.argmax(scores))]

    def update(self, task: Task, loss: float):
        if task.routed_gpu is None:
            return
        k = int(task.routed_gpu)
        if self.d is None or k < 0 or k >= len(self.A):
            return
        if task.bandit_context is None:
            return

        x = np.asarray(task.bandit_context, dtype=float).reshape(-1)
        if x.size != self.d:
            return

        r = -float(loss)
        self.A[k] += np.outer(x, x)
        self.b[k] += r * x


# -------------------- Registry with tunable hyperparams --------------------
# We keep the original ROUTING_REGISTRY interface (callable(V)->policy) intact,
# but also add a new helper: make_routing_policy(name, V, **kwargs)
# so run_experiment.py can pass bandit hyperparams without breaking old code.

def make_routing_policy(name: str, V: float, **kwargs) -> RoutingPolicy:
    name = str(name).lower().strip()
    if name == "arc":
        return ARCRouting(V=V)
    if name == "random":
        return RandomRouting()
    if name == "rr":
        return RoundRobinRouting()
    if name == "jsq":
        return JSQRouting()
    if name == "jsw":
        return JSWRouting()

    # bandits (tunable)
    if name == "mab_ucb1":
        c = float(kwargs.get("bandit_ucb_c", kwargs.get("c", 1.0)))
        return UCB1BanditRouting(c=c)

    if name == "cb_linucb":
        ucb_alpha = float(kwargs.get("linucb_alpha", kwargs.get("ucb_alpha", 1.0)))
        ridge = float(kwargs.get("linucb_ridge", kwargs.get("ridge", 1.0)))
        include_bias = bool(kwargs.get("linucb_bias", kwargs.get("include_bias", True)))
        return LinUCBContextualRouting(ucb_alpha=ucb_alpha, ridge=ridge, include_bias=include_bias)

    raise ValueError(f"Unknown routing policy: {name}")


ROUTING_REGISTRY: Dict[str, Callable[..., RoutingPolicy]] = {
    "arc": lambda V: ARCRouting(V=V),
    "random": lambda V: RandomRouting(),
    "rr": lambda V: RoundRobinRouting(),
    "jsq": lambda V: JSQRouting(),
    "jsw": lambda V: JSWRouting(),

    # default hyperparams (still works if you don't pass kwargs)
    "mab_ucb1": lambda V: UCB1BanditRouting(c=1.0),
    "cb_linucb": lambda V: LinUCBContextualRouting(ucb_alpha=1.0, ridge=1.0, include_bias=True),
}


class Dispatcher:
    """
    Clean dispatcher: no cost_mode branches.
    """
    def __init__(
        self,
        env: simpy.Environment,
        gpu_queues: List[GPUQueue],
        logger: GlobalEventLogger,
        routing_policy: RoutingPolicy,
    ):
        self.env = env
        self.queues = list(gpu_queues)
        self.logger = logger
        self.policy = routing_policy

    def route(self, task: Task):
        q = self.policy.select_queue(self.queues)

        # store routing decision and (optionally) bandit context on the task
        task.routed_gpu = int(q.idx)
        task.route_time_Q = float(len(q.queue))  # C2: queue length at route time

        if hasattr(self.policy, "compute_context"):
            ctx = self.policy.compute_context(q)
            if ctx is not None:
                task.bandit_context = np.asarray(ctx, dtype=float)

        self.logger.record_arrival(q.idx)
        q.enqueue(task)


class TaskGenerator:
    """
    Poisson arrivals (baseline synthetic workload).
    """
    def __init__(self, env: simpy.Environment, dispatcher: Dispatcher, arrival_rate: float, duration: float):
        self.env = env
        self.dispatcher = dispatcher
        self.arrival_rate = float(arrival_rate)
        self.duration = float(duration)
        self.tasks = []
        self.action = env.process(self.run())

    def run(self):
        while float(self.env.now) < self.duration:
            dt = np.random.exponential(1.0 / self.arrival_rate) if self.arrival_rate > 0 else 1e9
            yield self.env.timeout(float(dt))
            task = Task(float(self.env.now))
            self.tasks.append(task)
            self.dispatcher.route(task)


class TimestampTaskGenerator:
    """
    Trace-driven arrivals using an increasing array arrival_times.
    """
    def __init__(
        self,
        env: simpy.Environment,
        dispatcher: Dispatcher,
        arrival_times: List[float],
        duration: Optional[float] = None,
    ):
        self.env = env
        self.dispatcher = dispatcher
        self.arrival_times = [float(t) for t in arrival_times]
        self.duration = float(duration) if duration is not None else None
        self.tasks = []
        self.action = env.process(self.run())

    def run(self):
        if len(self.arrival_times) == 0:
            return

        t_first = self.arrival_times[0]
        if self.duration is not None and t_first > self.duration:
            return

        if t_first > 0:
            yield self.env.timeout(t_first)

        task = Task(float(self.env.now))
        self.tasks.append(task)
        self.dispatcher.route(task)

        for t in self.arrival_times[1:]:
            if self.duration is not None and t > self.duration:
                break
            now = float(self.env.now)
            dt = max(0.0, float(t) - now)
            if dt > 0:
                yield self.env.timeout(dt)

            task = Task(float(self.env.now))
            self.tasks.append(task)
            self.dispatcher.route(task)

