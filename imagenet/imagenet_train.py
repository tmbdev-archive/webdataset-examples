# -*- Python -*-

# A simple example of using WebDataset for ImageNet training.
# This uses the PyTorch Lightning framework.

import argparse
import collections
import logging

import torch
import torch.utils.data as data
from torch import nn
import torchvision

import pytorch_lightning as pl
import webdataset as wds

parser = argparse.ArgumentParser("ImageNet training on shards")
parser.add_argument("--shards", default="./data", help="shard directory")
parser.add_argument("--epoch", type=int, default=1000000, help="epoch length")
parser.add_argument("--batch-size", type=int, default=128, help="batch size")
parser.add_argument("--num-workers", type=int, default=8, help="dataloader workers")
parser.add_argument("--learning-rate", type=float, default=1e-3, help="learning rate")
parser.add_argument("--model", default="resnet18", help="desired torchvision model")
parser.add_argument(
    "--imagenet",
    help="if given, points to ImageNet directory and uses torchvision.datasets.ImageNet",
)
args = parser.parse_args()

# for convenience, we allow pathname specs for non-pipe: arguments
if args.shards.startswith("pipe:"):
    trainurls = args.shards % "imagenet-train-{000000..000146}.tar"
    valurls = args.shards % "imagenet-train-{000000..000006}.tar"
else:
    trainurls = args.shards.rstrip("/") + "/imagenet-train-{000000..000146}.tar"
    valurls = args.shards.rstrip("/") + "/imagenet-train-{000000..000006}.tar"

print("train:", trainurls)
print("val:", valurls)

image_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(0.3, 0.3, 0.3),
        torchvision.transforms.ToTensor(),
    ]
)


class Net(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.total = 0
        self.model = eval("torchvision.models.%s" % hparams.model)()
        self.criterion = nn.CrossEntropyLoss()
        self.errs = []

    def train_dataloader(self):

        # This "if" statement is the only difference between
        # WebDataset and torchvision.datasets.ImageNet
        if args.imagenet in [None, ""]:
            dataset = (
                wds.Dataset(trainurls)
                .shuffle(5000)
                .decode("pil")
                .to_tuple("ppm;jpg;jpeg;png", "cls")
                .map_tuple(image_transform, lambda x: x)
            )
            num_batches = (args.epoch + args.batch_size - 1) // args.batch_size
            dataset = wds.ChoppedDataset(dataset, args.epoch, nominal=num_batches)
        else:
            dataset = torchvision.datasets.ImageNet(
                args.imagenet, split="train", transform=image_transform
            )
            dataset = wds.ChoppedDataset(dataset, args.epoch)

        loader = data.DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )
        return loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=hparams.learning_rate)

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        self.total += len(inputs)
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, targets)
        errs = float((outputs.argmax(dim=1) != targets).sum() * 1.0 / len(targets))
        logs = {"train/loss": loss, "train/err": errs}
        return dict(loss=loss, log=logs)


# This is more-or-less standard PyTorch Lightning for model instantiation
# and training.
Hparams = collections.namedtuple("Hparams", "model learning_rate".split())
hparams = Hparams(args.model, args.learning_rate)
print("hparams", hparams)
net = Net(hparams)
trainer = pl.Trainer(gpus=[0])
logging.getLogger("lightning").setLevel(logging.WARNING)
trainer.fit(net)
