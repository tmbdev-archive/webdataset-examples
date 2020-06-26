# -*- Python -*-

# A simple example of using WebDataset for ImageNet training.
# This uses the PyTorch Lightning framework.

import sys
import argparse
import logging

import torch
import torch.utils.data as data
from torch import nn
import torchvision

import pytorch_lightning as pl
import webdataset as wds

import pickle


image_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(0.3, 0.3, 0.3),
        torchvision.transforms.ToTensor(),
    ]
)


def identity(x):
    return x


class Net(pl.LightningModule):
    def __init__(
        self,
        hparams,
        imagenet=None,
        trainurls=None,
        epoch=1000000,
        batch_size=64,
        num_workers=4,
    ):
        super().__init__()
        self.hparams = hparams
        self.imagenet = imagenet
        self.trainurls = trainurls
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epoch = epoch
        self.total = 0
        self.model = eval("torchvision.models.%s" % self.hparams["model"])()
        pickle.dumps(self.model)
        self.criterion = nn.CrossEntropyLoss()
        self.errs = []
        self.first_run = True

    def train_dataloader(self):

        # This "if" statement is the only difference between
        # WebDataset and torchvision.datasets.ImageNet
        if self.imagenet in [None, ""]:
            dataset = (
                wds.Dataset(self.trainurls, handler=wds.warn_and_continue)
                .shuffle(5000)
                .decode("pil", handler=wds.warn_and_continue)
                .to_tuple("ppm;jpg;jpeg;png", "cls")
                .map_tuple(image_transform, identity)
            )
            num_batches = (self.epoch + self.batch_size - 1) // self.batch_size
            dataset = wds.ResizedDataset(dataset, self.epoch, nominal=num_batches)
        else:
            dataset = torchvision.datasets.ImageNet(
                self.imagenet, split="train", transform=image_transform
            )
            dataset = wds.ResizedDataset(dataset, self.epoch)

        loader = data.DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])

    def forward(self, inputs):
        return self.model(inputs).type_as(inputs).to(self.device)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        self.total += len(inputs)
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, targets.to(device=outputs.device))
        errs = float(
            (outputs.detach().cpu().argmax(dim=1) != targets.detach().cpu()).sum()
            * 1.0
            / len(targets)
        )
        logs = {"train/loss": loss, "train/err": errs}
        if self.first_run:
            try:
                print(
                    "backend:",
                    torch.distributed.get_backend(),
                    "rank/size:",
                    torch.distributed.get_rank(),
                    torch.distributed.get_world_size(),
                    file=sys.stderr,
                )
            except Exception as exn:
                print(exn, sys.stderr)
            self.first_run = False
        return dict(loss=loss, log=logs)


def main(the_args, **kw):
    global args
    args = the_args
    hparams = dict(model=args.model, learning_rate=args.learning_rate)
    print("hparams", hparams)
    net = Net(hparams, **kw)
    kw = eval("dict(" + args.trainer + ")")
    print("Trainer config:", kw)
    trainer = pl.Trainer(**kw)
    logging.getLogger("lightning").setLevel(logging.WARNING)
    trainer.fit(net)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ImageNet training on shards")
    parser.add_argument("--shards", default="./data", help="shard directory")
    parser.add_argument("--epoch", type=int, default=1000000, help="epoch length")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="dataloader workers")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="learning rate"
    )
    parser.add_argument("--model", default="resnet50", help="desired torchvision model")
    parser.add_argument(
        "--trainer", default="gpus=[0]", help="extra arguments to trainer"
    )
    parser.add_argument(
        "--imagenet",
        help="if given, points to ImageNet directory and uses torchvision.datasets.ImageNet",
    )
    args = parser.parse_args()
    # for convenience, we allow pathname specs for non-pipe: arguments
    if args.shards.startswith("pipe:"):
        trainurls = args.shards % "imagenet-train-{000000..000146}.tar"
        valurls = args.shards % "imagenet-val-{000000..000006}.tar"
    else:
        trainurls = args.shards.rstrip("/") + "/imagenet-train-{000000..000146}.tar"
        valurls = args.shards.rstrip("/") + "/imagenet-val-{000000..000006}.tar"

    print("train:", trainurls)
    print("val:", valurls)

    main(
        args,
        trainurls=trainurls,
        epoch=args.epoch,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
