# Configuration options for the ./run script.
# You only need to change these if you are using Docker.
# If you just want to run the Python scripts directly,
# configure the paths with --imagenet and --shards

# path to the raw ImageNet dataset (suitable for use with
# torchvision.datasets.ImageNet)
imagenet="/mdata/imagenet-raw"

# path to the output directory where shards will be written
shards="$(pwd)/data"
