import time
import argparse
import os

import torchvision

import loaders
import trainer
import slog


parser = argparse.ArgumentParser("ImageNet Training")
parser.add_argument("--learning-rate", type=float, default=0.1)
parser.add_argument("--learning-schedule", type=int, default=30)
parser.add_argument("--epochs", type=int, default=120)
parser.add_argument("--model", default="resnet18")
parser.add_argument("--loaderargs", default="")
parser.add_argument("--valloaderargs", default="")
args = parser.parse_args()


logger = slog.Logger()
logger.json("__args__", args.__dict__)


def logbatch(trainer):
    entry = trainer.log[-1]
    entry["lr"] = trainer.last_lr
    logger.json("batch", trainer.log[-1], step=trainer.total)


def schedule(total):
    epoch = total // 1000000
    return args.learning_rate * (0.1 ** (epoch // args.learning_schedule))


model = eval(f"torchvision.models.{args.model}()").cuda()
trainer = trainer.Trainer(model, schedule=schedule)
trainer.after_batch = logbatch

loader = loaders.make_train_loader(**eval(f"dict({args.loaderargs})"))
val_loader = loaders.make_val_loader(**eval(f"dict({args.valloaderargs})"))

for epoch in range(args.epochs):
    trainer.train_epoch(loader)
    loss, err = trainer.errors(val_loader)
    print("test", trainer.total, loss, err)
    logger.add_scalar("val/loss", loss, trainer.total)
    logger.add_scalar("val/top1", err, trainer.total)
    logger.save("model", model, trainer.total)
