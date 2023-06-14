from dataclasses import dataclass
from typing import Dict


@dataclass
class BestResult:
    avg_acc_obs: float = 0
    avg_acc_chunk: float = 0
    avg_f1_chunk: float = 0
    avg_loss: float = -float("inf")
    epoch: int = 0

    def to_dict(self) -> Dict:
        data = {"avg_acc_obs": self.avg_acc_obs,
                "avg_acc_chunk": self.avg_acc_chunk,
                "avg_f1_chunk": self.avg_f1_chunk,
                "avg_loss": self.avg_loss,
                "epoch": self.epoch}
        return data
