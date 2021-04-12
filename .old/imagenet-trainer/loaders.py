import torch.utils.data as data
import torchvision
import webdataset as wds


trainurls = "./shards/imagenet-train-{000000..001281}.tar"
valurls = "./shards/imagenet-val-{000000..000049}.tar"


def identity(x):
    return x


normalize = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


def make_train_loader(epoch_size=1100000, batch_size=64, shuffle=20000):

    # num_batches = (epoch_size + batch_size - 1) // batch_size

    if True:
        image_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        image_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                normalize,
            ]
        )

    dataset = (
        wds.Dataset(trainurls, handler=wds.warn_and_stop, length=epoch_size)
        .shuffle(shuffle)
        .decode("pil", handler=wds.warn_and_continue)
        .to_tuple("ppm;jpg;jpeg;png", "cls", handler=wds.warn_and_continue)
        .map_tuple(image_transform, identity, handler=wds.warn_and_continue)
        .batched(batch_size)
    )

    loader = data.DataLoader(dataset, batch_size=None, num_workers=4)
    return loader


def make_val_loader(epoch_size=50000, batch_size=64):
    val_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize,
        ]
    )

    val_dataset = (
        wds.Dataset(valurls, handler=wds.warn_and_stop, prepare_for_worker=False)
        .decode("pil", handler=wds.warn_and_continue)
        .to_tuple("ppm;jpg;jpeg;png", "cls", handler=wds.warn_and_continue)
        .map_tuple(val_transform, identity, handler=wds.warn_and_continue)
        .batched(batch_size)
    )
    val_loader = data.DataLoader(val_dataset, batch_size=None, num_workers=4)
    return val_loader
