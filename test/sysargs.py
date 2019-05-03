import sys

args = sys.argv

dataset = str(args[1])
size = int(args[2])
split = bool(args[3])

print(dataset, size, split)

