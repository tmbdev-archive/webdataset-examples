import sys
import time

import numpy as np
import torch
from torch import nn
import importlib.util


class Struct:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class LRConstant(object):
    def __init__(self, lr):
        self.lr = lr

    def __call__(self, n):
        return self.lr


class LRSchedule(object):
    def __init__(self, schedule):
        assert isinstance(schedule, list)
        assert len(schedule[0]) == 2
        assert isinstance(schedule[0][0], int)
        assert isinstance(schedule[0][1], float)
        self.schedule = schedule

    def __call__(self, n):
        last = self.schedule[0][1]
        for i, r in self.schedule:
            if n <= i:
                break
            last = r
        return last


def identity(x):
    return x


class Every(object):
    def __init__(self, t):
        self.t = t
        self.last = time.time()

    def __call__(self):
        now = time.time()
        if now - self.last >= self.t:
            self.last = now
            return True
        return False


class Trainer(object):
    def __init__(
        self,
        model,
        criterion=nn.CrossEntropyLoss(),
        device="cuda:0",
        progress=True,
        report=10.0,
        schedule=LRConstant(0.01),
        momentum=0.9,
        weight_decay=1e-4,
        world_size=1,
    ):
        self.device = device
        self.model = model
        self.model.to(self.device)
        self.criterion = criterion
        self.progress = progress
        self.report = report
        self.world_size = world_size
        self.total = 0
        self.log = []
        self.last_lr = -1
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.set_schedule(schedule)
        self.after_batch = None

    def set_schedule(self, lr):
        assert callable(lr)
        self.schedule = lr
        return self

    def _set_lr(self, lr):
        if lr == self.last_lr:
            return self
        print(f"setting learning rate to {lr}", file=sys.stderr, flush=True)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.last_lr = lr
        return self

    def batch_errors(self, outputs, targets):
        pred = outputs.detach().cpu().argmax(dim=1)
        targets = targets.detach().cpu()
        return float((pred != targets).sum()) / len(targets)

    def train_batch(self, inputs, targets):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(inputs.to(self.device))
        loss = self.criterion(outputs, targets.to(device=outputs.device))
        loss.backward()
        self.optimizer.step()
        err = self.batch_errors(outputs, targets)
        self.total += len(inputs) * self.world_size
        self.log.append(dict(loss=float(loss), err=float(err), count=self.total))

    def train_epoch(self, loader):
        self.model.train()
        report = Every(self.report)
        for inputs, targets in loader:
            self._set_lr(self.schedule(self.total))
            self.train_batch(inputs, targets)
            if self.after_batch is not None:
                self.after_batch(self)
            if self.progress and report():
                r = self.log[-1]
                info = f"{r['count']:12d} loss={r['loss']:.3f} err={r['err']:.3f}"
                print(info, end="        \r", file=sys.stderr, flush=True)

    def errors(self, loader, nval=100000):
        losses, errs = [], []
        self.model.eval()
        total = 0
        with torch.no_grad():
            for inputs, targets in loader:
                if total >= nval:
                    break
                outputs = self.model(inputs.to(self.device))
                loss = self.criterion(outputs, targets.to(device=outputs.device))
                err = self.batch_errors(outputs, targets)
                losses.append(float(loss))
                errs.append(float(err))
                total += len(outputs)
        return np.mean(losses), np.mean(errs)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
