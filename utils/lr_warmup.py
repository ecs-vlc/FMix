from torchbearer.callbacks import Callback
import torchbearer


class WarmupLR(Callback):
    def __init__(self, min_lr, max_lr, warmup_period=5):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.t = warmup_period

    def on_start_training(self, state):
        super().on_start_training(state)
        if state[torchbearer.EPOCH] < self.t:
            delta = (self.t - state[torchbearer.EPOCH])/self.t
            opt = state[torchbearer.OPTIMIZER]

            for pg in opt.param_groups:
                pg['lr'] = self.min_lr * delta + self.max_lr * (1-delta)
