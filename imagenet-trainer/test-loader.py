import trainer
import time
import argparse


parser = argparse.ArgumentParser("test loader")
parser.add_argument("--loaders", default="./loaders.py")
parser.add_argument("--extras", default="")
parser.add_argument("--which", default="train")
parser.add_argument("--start", type=int, default=20)
parser.add_argument("--nbatches", type=int, default=40)
args = parser.parse_args()


loaders = trainer.load_module("loaders", args.loaders)
kw = eval(f"dict({args.extras})")
loader = eval(f"loaders.make_{args.which}_loader")(**kw)

total = 0
source = iter(loader)
count = 0


while count < args.start:
    next(source)
    count += 1


start = time.time()
while count < args.start+args.nbatches:
    batch = next(source)
    total += len(batch[0])
    print(total, end="    \r", flush=True)
    count += 1
print()
end = time.time()


print("samples/s", total / (end - start))

for i, t in enumerate(batch):
    print(i, type(t), getattr(t, "shape"), getattr(t, "dtype"))
