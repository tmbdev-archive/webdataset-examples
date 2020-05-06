# A small demonstration of using WebDataset with ImageNet.

This is a small repo illustrating how to use WebDataset on ImageNet.
There are environments in which this code is useful and leads
to speedups, but its primary purpose is just to give an example
of WebDataset on a familiar problem.

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

```



```


# Other Runtime Environments

## Starting a Web Server and Training

Training against a web server is easy. Let's start by starting up a web server;
here `$shards` is the directory containing the ImageNet shards you generated
above.

```Shell
$ docker run --name nginx-data --rm -v $shards:/usr/share/nginx/html:ro -p 8080:80 nginx
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
