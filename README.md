# PyTorch Imagenet Example

This is a minimal modification of the PyTorch Imagenet example; it is not the best example to base your code on because the PyTorch Imagenet example itself is fairly old code.

Please go to 

http://github.com/webdataset

for up-to-date examples of WebDataset usage.


```






```


# Introduction

This code is as close as possible to the original PyTorch example to illustrate
the changes necessary to move from PyTorch indexed datasets to iterable datasets.
The original example is quite complex because of all the different options.
This code inherits that complexity.

For new developments, you may want to start with [tmbdev/webdataset-lightning](http://github.com/tmbdev/webdataset-lightning) or other examples.

# VirtualEnv

Before you do anything, set up a virtualenv with

```Bash
$ ./run venv
```

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
$ ./run makeshards
```

The `./run` command is just a helper script that will run Python commands
for you in a virtual environment.

This should take a few minutes, and eventually, you should end up with 490
training shards and 23 validation shards. You can split your dataset into larger
or smaller shards, depending on your needs. These shards happen to contain
300 Mbytes worth of examples each.

Have a look at `makeshards.py` to see how the shards are written.

# Running Training

After generating the shards, you can run training with:

```Bash
    $ ./run single  # DataParallel training
```


```Bash
    $ ./run multi # DistributedDataParallel training
```

The `./run multi` command uses functionality in the original example
that spawns multiple subprocesses on the same machine and illustrates
the functionality. If you want to run truly distributed processes,
this should work the same way, but you need to start up the processes
in whatever way you are used to starting them.

For true distributed training, it is most convenient to put the
dataset on some web server. In the cloud, you can train from S3 or GCS
directly.

# Changes From Original

The changes from the original are:

- creation of loaders has been refactored into separate functions
  (`make_train_loader_orig` and `make_train_loader_wds`)
- the original code called `dataset.sampler.set_epoch`, but that method
  is not available on iterable datasets, so the call is conditional
- for the original data loader, the directory is now an option (rather
  than a positional argument)
- a few new command line options

You can see the differences by typing:

```Bash
$ diff main-orig.py main-wds.py
```

or

```Bash
$ meld main-orig.py main-wds.py
```
