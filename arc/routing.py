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

