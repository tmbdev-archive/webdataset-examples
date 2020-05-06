import logging
import argparse

import torch
import torch.utils.data as data
from torch import nn
from torchvision import models, transforms

import pytorch_lightning as pl
import webdataset as wds

parser = argparse.ArgumentParser("ImageNet training on shards")
parser.add_argument("--shards", default="./data")
parser.add_argument("--epoch", type=int, default=1000000)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--num-workers", type=int, default=8)
args = parser.parse_args()

if args.shards.startswith("pipe:"):
    trainurls = args.shards % "imagenet-train-{000000..000146}.tar"
    valurls = args.shards % "imagenet-train-{000000..000006}.tar"
else:
    trainurls = args.shards.rstrip("/") + "/imagenet-train-{000000..000146}.tar"
    valurls = args.shards.rstrip("/") + "/imagenet-train-{000000..000006}.tar"

print("train:", trainurls)
print("val:", valurls)

image_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(),
    ]
)


class Net(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.total = 0
        self.model = models.resnet18()
        self.criterion = nn.CrossEntropyLoss()
        self.errs = []

    def train_dataloader(self):
        dataset = (
            wds.Dataset(trainurls)
            .shuffle(5000)
            .decode("pil")
            .to_tuple("ppm;jpg;jpeg;png", "cls")
            .map_tuple(image_transform, lambda x: x)
        )
        num_batches = (args.epoch + args.batch_size - 1) // args.batch_size
        dataset = wds.ChoppedDataset(dataset, args.epoch, nominal=num_batches)
        loader = data.DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )
        return loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

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


net = Net()
trainer = pl.Trainer(gpus=[0])

logging.getLogger("lightning").setLevel(logging.WARNING)
trainer.fit(net)
