# Introduction

This is a derivative of the PyTorch Imagenet training example from the
[PyTorch examples folder](https://github.com/pytorch/examples/tree/master/imagenet).

This is mainly aimed at people familiar with that example; for new code, you may
want to use the `imagenet-lightning` example in this tree (which uses the
PyTorch Lightning framework) or the `simple-imagenet` example, which organizes
training a bit better.

The original code is in `main-orig.py` and the WebDataset-based code is in
`main-wds.py`; the latter lets you switch between WebDataset and file based
loading using a command line switch.

# Generating the Shards

Before training with WebDataset, you need a sharded dataset. Datasets are becoming
available directly in WebDataset format, but since Imagenet is not freely
redistributable, you have to generate the WebDataset version of Imagenet yourself
from your Imagenet distribution. The script `makeshards.py` will do this for you.

Let's say you have installed ImageNet in `/data/imagenet` so that you can
train with `torchvision.datasets.ImageNet("/data/imagenet")`. To transform
that data into shards, you can use:
```Bash
$ ln -s /data/imagenet ./data
$ mkdir ./shards
$ ./run makeshards.py
```

The `./run` command is just a helper script that will run Python commands
for you in a virtual environment.

This should take a few minutes, and eventually, you should end up with 1282
training shards and 50 validation shards. You can split your dataset into larger
or smaller shards, depending on your needs. These shards happen to contain
1000 training exaamples each.

Have a look at `makeshards.py` to see how the shards are written.

# Running Training

You can simply run training using the original and the new data set implementations
using `./run bench`.

Have a look at `main-wds.py` to see the different options for selecting different
dataset implementations, different augmentation methods

# Changes From Original

The changes from the original are:

- creation of loaders has been refactored into separate functions
  (`make_train_loader_orig` and `make_train_loader_wds`)
- the original code called `dataset.sampler.set_epoch`, but that method
  is not available on iterable datasets, so the call is conditional
- for the original data loader, the directory is now an option (rather
  than a positional argument)
- a few new command line options
