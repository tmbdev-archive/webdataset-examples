# General Comments

    WebDataset doesn't randomly distribute samples across nodes, but shards. For large training jobs and many nodes, you should split your dataset into a few hundred shards at least.
    For IterableDatasets, you should do batching inside the Dataset so that whole batches are transferred to the DataLoader; this isn't specific to WebDataset. However, if you want to mix data more, you can unbatch, shuffle, and rebatch in the training job (MultiDataset makes that particularly simple, but you can also do it with DataLoader)
    Which shards are selected is handled by three hooks in WebDataset:
        the reseed_hook is called allowing you to reseed the random number generator (default: noop)
        the node_selection hook is called on a list of shards, returning the node-specific subset (default: noop)
        the shard_selection hook is called on that list of shards, returning the final subset of shards
        the shard_shuffle hook is called to shuffle those shards

By default, training jobs on each node will see the entire dataset for training; that's because distributed frameworks have difficulties with different numbers of training batches from different nodes. If you don't want that behavior, you need to update the node_selection hook to select a subset of shards for each node.

This setup attempts to satisfy a number of different existing constraints as well as possible: the expectations of distributed frameworks, the constraint that distributed sequential I/O cannot guarantee exactly equal distribution of samples, and the existing APIs.

(For precise epochs and equal number of batches per worker, you need additional network communications somewhere; Tensorcom with PUBSUB sockets are a cleaner and more efficient solution.)

**NB: this example needs updating and may not work with the latest versions of Lightning and WebDataset**

# A small demonstration of using WebDataset with ImageNet and PyTorch Lightning

This is a small repo illustrating how to use WebDataset on ImageNet.
using the PyTorch Lightning framework.

# Python

There are two Python programs:

- `imagenet_shards.py` converts the ImageNet data from file
  format to WebDataset format; it will create 147 `.tar` files
  containing all the images and their associated classes
- `imagenet_train.py` performs training against the shards

These are written in Python3 and use PyTorch Lightning as the framework.

If you have all the requirements installed (`pip3 install -r requirements.txt`),
you can directly use these scripts for training. You do need to tell
it where the ImageNet data directory is and where the shards should go.

Install the requirements first (you may want to use a virtualenv):

```Shell
pip3 install -r requirements.txt
```

Next, convert ImageNet to sharded format. The `--imagenet=` argument
needs to point to the original ImageNet tree (the same tree that
`torchvision.datasets.ImageNet` expects). The `--shards=` argument
tells the command where to put the output shards; the default is
`./data`:

```Shell
python3 ./imagenet_shards.py --imagenet=...
```

Next, you can train against the shards you just generated (use
the `--shards=` argument again if you put them somewhere else).

```Shell
python3 ./imagenet_train.py
```

That's all you need.

You can use the traditional `torchvision.datasets.ImageNet` loader
using

```Shell
python3 ./imagenet_train.py --imagenet=/root/of/imagenet
```

If ImageNet is stored on rotational drives, this is about 30-40% slower
than WebDataset. On NVMe drives, you shouldn't see much of a difference.

Have a look at what changes you need to make in order to use WebDataset;
they are minimal.

Note that `imagenet_train.py` uses `ChoppedDataset` to allow the epoch size
and nominal size to be adjusted. Adjustment of the epoch size is often convenient.
Adjustment of the nominal size is currently required due to the odd semantics
of `DataLoader` for `IterableDataset.__len__`; an issue has been opened that
will hopefully make this unnecessary.

# DistributedDataParallel


You can run distributed data parallel jobs with something like:

```Shell
host0$ NODE_RANK=0 MASTER_ADDR=host0 MASTER_PORT=11111 python3 imagenet_train.py --trainer='gpus=1,nb_gpu_nodes=2,distributed_backend="ddp"'
...
host1$ NODE_RANK=1 MASTER_ADDR=host0 MASTER_PORT=11111 python3 imagenet_train.py --trainer='gpus=1,nb_gpu_nodes=2,distributed_backend="ddp"'
...

```

Note that the `--trainer` arguments are evaluated and interpolated into the `Trainer` arguments.
Other useful arguments are:
- `replace_sampler_ddp=False`
- `auto_lr_find=True`


# Other Runtime Environments

This section is not so much about what you need to do in order to use
WebDataset, but about all the different things it enables.

TL;DR is this: with WebDataset, you can pretty much train from any data source
by using a `pipe:` URL; for example:

- train over HTTP
    - `python3 imagenet_train.py --base='pipe:curl -s http://localhost:8080/'`
- train from Google Cloud Bucket
    - `python3 imagenet_train.py --base='pipe:gsutil cat gs://bucket/%s'`
- train from Azure Container
    - `python3 imagenet_train.py --base='pipe:az storage blob download ... -f /dev/stdout`'


(Note that the `%s` convention here for combining the shard spec with
the storage bucket is just implemented by `imagenet_train.py`; it is
not a part of WebDataset.)

## Optimizing Local Disk Access

If your data is stored on a single rotational drive but you use
multiple `DataLoader` processes, the reads from the different processes
will interfere with one another and give you less I/O bandwidth than you
would otherwise get. Of course, you could reduce `num_workers` (`--num-workers`),
but in that case, you may not have enough CPU for the data augmentation.
Generally, if you do nothing, `num_workers=8` will result in about 120 Mbytes/s
I/O bandwidth on a rotational drive.

`WebDataset` gives you a couple of options for addressing this case:

- You can use the `GOPEN_BUFFER` environment variable to increase the
  buffer size in the streams opened by `WebDataset`

- You can use an existing buffering program like `mbuffer` to buffer
  large reads, e.g., `--shards='pipe:mbuffer /my/shards/%s'`

- You can use a custom program that serializes reads, like `fread`
  included here, and then use `--shards='pipe:./fread /my/shards/%s'`

With one of these, you can usually increase aggregate I/O bandwidth from a
single rotational drive to about 200 Mbytes/s even using multiple workers trying
to read simultaneously.

Mechanisms like these can be even more important when your primary storage is a
RAID system.


## Starting a Web Server and Training

Training against a web server is easy. Let's start by starting up a web server;
here `$shards` is the directory containing the ImageNet shards you generated
above.

```Shell
$ docker run --name nginx-data --rm -v $shards:/usr/share/nginx/html:ro -p 8080:80 nginx
```

You can also use the built-in web server:

```Shell
$ python3 -m http.server --dir=$shards 8080
```

Now let's train against it:

```Shell
$ python3 imagenet_train.py --base='pipe:curl -s http://localhost:8080/'
...
```

Of course, you can also train remotely from any other machine that
can reach the web server

## Training against a Cloud Bucket

If your data is stored in the cloud, you can use what ever cloud
command line you like for reading from the cloud server. For
Google Cloud Storage, this looks as follows:

```Shell
$ python3 imagenet_train.py --base='pipe:gsutil cat gs://bucket/%s`
```

This is particularly useful when running training jobs in the cloud
inside a virtual machine or a cloud-hosted Kubernetes.

## Docker

If you prefer, you can also run everything in a Docker container.
The `./run` script helps with that. The configuration is in
`config.sh`; you need to specify the directory containing
the raw ImageNet data and the directory where the shards go.

```Shell
$ vim config.sh                  # configure the paths
...
$ ./run build                    # builds the container
...
$ ./run imagenet_shards.py       # converts from file-based Imagenet to shards
# writing /data/imagenet-train-000000.tar 0 0.0 GB 0
# writing /data/imagenet-train-000001.tar 8735 1.0 GB 8735
# writing /data/imagenet-train-000002.tar 8804 1.0 GB 17539
# writing /data/imagenet-train-000003.tar 8771 1.0 GB 26310
...
$ ./run imagenet_train.py        # run the training code
...
```

Since ImageNet fits entirely on a typical NVMe drive, you will
probably not see much of a performance difference between regular the
`FileDataset`-based code and the `WebDataset`-based code when
training from an NVMe drive.

If you store the datasets on a rotational drive, the WebDataset-based
code can be a factor of two faster.

## Kubernetes or Cloud

There are many different ways you can run Kubernetes, and usually, you'll
want to run it in the cluster or in the cloud.

For example, if you use Google Kubernetes Engine, you need to push your
Docker image first and probably want to copy the shards to your cloud
bucket. Afterwards, you can run training in the cloud with a command like:

```Shell
$ image=grc.io/my/img shards='pipe:gsutil cat gs://bucket/%s' envsubst < ./train.yml |
kubectl apply -f
```

There are many possible configurations, storage options, image repos, etc.


