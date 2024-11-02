import onnx 
import argparse

parser = argparse.ArgumentParser(description='Check model inputs')
parser.add_argument('model', type=str, help='Model path')

args = parser.parse_args()

m = onnx.load(args.model, load_external_data=True)
for i in m.graph.input:
    print(i.name)