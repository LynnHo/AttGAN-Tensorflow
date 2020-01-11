class LinearDecayLR:
    # if `step` < `step_start_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    # `step` should start from 0(included) to `steps`(excluded)

    def __init__(self, initial_learning_rate, steps, step_start_decay):
        self._initial_learning_rate = initial_learning_rate
        self._steps = steps
        self._step_start_decay = step_start_decay
        self.current_learning_rate = initial_learning_rate

    def __call__(self, step):
        if step >= self._step_start_decay:
            self.current_learning_rate = self._initial_learning_rate * (1 - 1 / (self._steps - self._step_start_decay + 1) * (step - self._step_start_decay + 1))
        else:
            self.current_learning_rate = self._initial_learning_rate
        return self.current_learning_rate


class StepDecayLR:

    def __init__(self, initial_learning_rate, step_size, decay_rate):
        super(StepDecayLR, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._step_size = step_size
        self._decay_rate = decay_rate
        self.current_learning_rate = initial_learning_rate

    def __call__(self, step):
        self.current_learning_rate = self._initial_learning_rate * self._decay_rate ** (step // self._step_size)
        return self.current_learning_rate
