import onnx 
import argparse

parser = argparse.ArgumentParser(description='Check model inputs')
parser.add_argument('model', type=str, help='Model path')

model = parser.parse_args().model

m = onnx.load(model, load_external_data=True)
for i in m.graph.input:
    print(i.name)