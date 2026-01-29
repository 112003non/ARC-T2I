# arc/batching.py
class BatchingPolicy:
    name = "base"

    def wait_before_service(self, q):
        return 0.0


class ARCBatching(BatchingPolicy):
    name = "arc"


class ThresholdBatching(BatchingPolicy):
    name = "threshold"

    def __init__(self, Wmax: float):
        self.Wmax = float(Wmax)

