# ImageNet with WebDataset and AIS

A sample project for training ImageNet models using WebDataset / AIStore.
Demonstrates multinode training on NGC Beta using AIStore for storage.

This code is intended for reading and understanding, not just turnkey running.

# WebDataset and DataLoader

## Single `DataLoader` Instance

When using `webdataset.Dataset` with a single `DataLoader` instance,
things are very simple: you create a dataset and then a loader,
with any number of workers you like. You can use single GPU
or multi-GPU training using `DataParallel`:

```Python
dataset = wds.Dataset(urls).shuffle(...) ... .to_tuple(...) ...
loader = data.DataLoader(dataset, batch_size=..., num_workers=...)
```

## Multinode Training

In multinode training with `DataLoader`, things become more complicated
because there are multiple input pipelines running in parallel
on different nodes.  Each node runs multiple Dataset instances in
subprocesses; the DataLoader collects the samples from subprocesses,
batches them up, and delivers the batch to the training loop running on
that node. Since there are multiple processes and nodes, this happens many
times in parallel.  This is inherently more complicated, no matter what
`Dataset` implementation you use. Old-style indexable datasets require
distributed samplers for this.  WebDataset gives you several options
for writing distributed input pipelines.

The way DistributedDataParallel works, it requires _exactly_ the same number
of batches to be delievered to each training loop, otherwise the job will
stall waiting for missing batches. Furthermore, `DataLoader` itself
(oddly enough) requires the `Dataset` to return the number of batches
as its length, not the number of samples per epoch.

We usually want data to be shuffled well during training. In addition,
for small datasets, we often want to make sure that each training sample
is used exactly once during an epoch, though for large and augmented datasets,
that is less of a concern.

By default, `webdataset.Dataset` will simply iterate through all the shards
it has, in order. Shards can be given either as a string with brace notation
or as a list of shard names. If you call the `.shuffle(N)` method, it will
shuffle both shards and samples.

If you arrange the number of shards to be divisible by the number of
`webdataset.Dataset` instances, give a subset of each shard to each instance,
and put the same number of samples in each shard, each `DataLoader` will
get the exact same number of samples. This is a lot of effort, though, and
not usually necessary. It also means that your job will fail if samples are
temporarily unavailable (e.g., due to temporary network errors). 

The `ResizedDataset` class is usually an easier and better way of
addressing all these constraints. A `ResizedDataset` will repeatedly
iterate through a given input dataset, with iterators of the desired epoch
length. In addition, it lets you specify a nominal epoch length different
from the actual epoch length.

A simple way of writing a distributed input pipeline is the following.
First, we need to calculate the epoch size per `Dataset` instance
and the number of batches:

```Python
num_dataset_instances = world_size * num_workers
epoch_size = datasets_size // num_dataset_instances
num_batches = (epoch_size + batch_size - 1) // batch_size
```

Next, dataset creation is almost identical to the simple case;
we just wrap a `ResizedDataset` instance around the original
dataset:

```Python
dataset = wds.Dataset(urls).shuffle(...) ... .to_tuple(...) ...
dataset = wds.ResizedDataset(dataset, epoch_size, nominal=num_batches)
loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=...)
```

This is used in the training data loader in the example.

There are several other options:

- You can let the `webdataset.Dataset` class perform the batching; this
  results in less shuffling, but is good for validation loops. This
  is used in the validation data loader in the example.
- You can split up the shard urls by node (`--shardsplit`).
- You can assign shards to datasets dynamically by dequeuing them from
  a shared queue.

# Files

- `run` -- script for building / running training
- `Dockerfile` -- container build file for local Docker jobs and NGC
- `imagenet_shards.py` -- script for converting ImageNet data to shards
- `train.py` -- main training script
- `config.sh` -- configuration options (used by `run`)
- `requirements.txt` -- Python packages used
- `no-key.json` -- empty key file (to make Docker ADD command work)

Helper scripts:

- `startnode_ais.sh` -- small helper script for running jobs on NGC
- `upload.sh` -- helper script for uploading results (you need to edit this!)

# `Run` Script

The `run` script lets you run things locally, in Docker, or on NGC.

You should start off with

```Shell
$ ./run local1
```

For running on NGC, you need to modify the `upload.sh` script somehow,
either to do nothing, or to upload data to a cloud provider or other
destination of our choice.

- `venv` -- set up a virtualenv
- `build` -- build the container
- `push` -- push the image to the cloud
- `shell` -- run shell or command in image container locally
- `local1` -- run a 1-gpu job in virtualenv
- `local2` -- run a 2-gpu job in virtualenv
- `docker2` -- run a 2 gpu job in docker
- `singlenode` -- run a single node job on NGC
- `ais` -- run a multinode job on NGC using AIS
- `tail` -- follow the output of the last NGC job
- `killall` -- kill all running NGC jobs

Run `python3 train.py` to get a list of training options.

# Additional Notes

- Have a look at the different `cmd_` functions in the `run` script
  to see how to run this code locally and distributed.
- The code is setup to use the `upload.sh` script to upload data to 
  somewhere. As distributed, `upload.sh` uploads to Google Cloud, but
  in order to make that work, you need to set up a Google Cloud bucket
  and a service account, and store the service account key in
  `google-key.json`. If you don't care about that, just edit `update.sh`
  to do nothing.
- Distributed training uses `torch.distributed.launch`
