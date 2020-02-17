from torchbearer import Callback


class MSDAAlternator(Callback):
    def __init__(self, msda_a, msda_b, n_a=1, n_b=1):
        super().__init__()
        self.augs = ((msda_a, n_a), (msda_b, n_b))
        self.current_aug = 0
        self.current_steps = 0

    def on_sample(self, state):
        super().on_sample(state)

        aug, steps = self.augs[self.current_aug]
        aug.on_sample(state)

        self.current_steps = self.current_steps + 1
        if self.current_steps >= steps:
            self.current_aug = (self.current_aug + 1) % 2
            self.current_steps = 0