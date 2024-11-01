from olive.common.utils import load_weights
import argparse

parser = argparse.ArgumentParser(description='Check adapter weights and shapes')
parser.add_argument('adapter1', type=str, help='First adapter')
parser.add_argument('adapter2', type=str, help='Second adapter')
args = parser.parse_args()

weights = load_weights(args.adapter1)
weights2 = load_weights(args.adapter2)


print("First model")
print(len(weights))
for key in sorted(weights):
#    assert weights[key].shape == weights2[key].shape
    print(key)

print("Second model")
print(len(weights2))
for key in sorted(weights2):
#    assert weights[key].shape == weights2[key].shape
    print(key)

for key in sorted(weights):
    assert weights[key].shape == weights2[key].shape
