# arc/routing.py
import numpy as np
from typing import List, Optional, Dict, Callable

from simulator.async_dispatch_simulator import (
    GPUQueue,
    Task,
)

# -------------------- Base --------------------

class RoutingPolicy:
    name = "base"

    def select_queue(self, queues: List[GPUQueue]) -> GPUQueue:
        raise NotImplementedError

    def compute_context(self, q: GPUQueue):
        return None

    def update(self, task: Task, loss: float):
        return


# -------------------- Deterministic policies --------------------

class ARCRouting(RoutingPolicy):
    name = "arc"

    def __init__(self, V: float):
        self.V = float(V)

    def select_queue(self, queues: List[GPUQueue]) -> GPUQueue:
        scores = []
        for q in queues:
            Q = len(q.queue)
            mu = q.hat_mu(Q)
            rho = q.rho(Q)
            scores.append(Q / mu + self.V * rho)
        return queues[int(np.argmin(scores))]


class RandomRouting(RoutingPolicy):
    name = "random"

    def select_queue(self, queues: List[GPUQueue]) -> GPUQueue:
        return np.random.choice(queues)


class RoundRobinRouting(RoutingPolicy):
    name = "rr"

    def __init__(self):
        self.i = 0

    def select_queue(self, queues: List[GPUQueue]) -> GPUQueue:
        q = queues[self.i % len(queues)]
        self.i += 1
        return q


class JSQRouting(RoutingPolicy):
    name = "jsq"

    def select_queue(self, queues: List[GPUQueue]) -> GPUQueue:
        return min(queues, key=lambda q: len(q.queue))


class JSWRouting(RoutingPolicy):
    name = "jsw"

    def select_queue(self, queues: List[GPUQueue]) -> GPUQueue:
        scores = []
        for q in queues:
            Q = len(q.queue)
            mu = max(q.hat_mu(Q), 1e-9)
            scores.append(Q / mu)
        return queues[int(np.argmin(scores))]


# -------------------- Bandits --------------------

class UCB1BanditRouting(RoutingPolicy):
    name = "mab_ucb1"

    def __init__(self, c: float = 1.0):
        self.c = float(c)
        self.counts = None
        self.means = None
        self.t = 0

    def _init(self, K):
        if self.counts is None:
            self.counts = np.zeros(K)
            self.means = np.zeros(K)

    def select_queue(self, queues):
        K = len(queues)
        self._init(K)
        self.t += 1

        for k in range(K):
            if self.counts[k] == 0:
                return queues[k]

        ucb = self.means + self.c * np.sqrt(np.log(self.t) / self.counts)
        return queues[int(np.argmax(ucb))]

    def update(self, task: Task, loss: float):
        if task.routed_gpu is None:
            return
        k = task.routed_gpu
        r = -loss
        n = self.counts[k]
        self.counts[k] += 1
        self.means[k] = (self.means[k] * n + r) / (n + 1)


class LinUCBContextualRouting(RoutingPolicy):
    name = "cb_linucb"

    def __init__(self, ucb_alpha=1.0, ridge=1.0, include_bias=True):
        self.ucb_alpha = ucb_alpha
        self.ridge = ridge
        self.include_bias = include_bias
        self.A = []
        self.b = []
        self.d = None

    def _context(self, q: GPUQueue):
        Q = len(q.queue)
        util = q.busy_time / q.env.now if q.env.now > 0 else 0.0
        lat = getattr(q, "recent_latency_ewma", 0.0)
        x = np.array([Q, util, lat])
        if self.include_bias:
            x = np.concatenate([[1.0], x])
        return x

    def select_queue(self, queues):
        K = len(queues)
        x0 = self._context(queues[0])
        if self.d is None:
            self.d = len(x0)
            self.A = [self.ridge * np.eye(self.d) for _ in range(K)]
            self.b = [np.zeros(self.d) for _ in range(K)]

        scores = []
        for k, q in enumerate(queues):
            x = self._context(q)
            Ainv = np.linalg.inv(self.A[k])
            theta = Ainv @ self.b[k]
            mean = theta @ x
            bonus = self.ucb_alpha * np.sqrt(x @ Ainv @ x)
            scores.append(mean + bonus)

        return queues[int(np.argmax(scores))]

    def compute_context(self, q: GPUQueue):
        return self._context(q)

    def update(self, task: Task, loss: float):
        if task.routed_gpu is None or task.bandit_context is None:
            return
        k = task.routed_gpu
        x = task.bandit_context
        r = -loss
        self.A[k] += np.outer(x, x)
        self.b[k] += r * x


# -------------------- Factory --------------------

def make_routing_policy(name: str, V: float, **kwargs) -> RoutingPolicy:
    name = name.lower()
    if name == "arc":
        return ARCRouting(V)
    if name == "random":
        return RandomRouting()
    if name == "rr":
        return RoundRobinRouting()
    if name == "jsq":
        return JSQRouting()
    if name == "jsw":
        return JSWRouting()
    if name == "mab_ucb1":
        return UCB1BanditRouting(kwargs.get("bandit_ucb_c", 1.0))
    if name == "cb_linucb":
        return LinUCBContextualRouting(
            kwargs.get("linucb_alpha", 1.0),
