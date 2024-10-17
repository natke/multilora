import onnx 

m = onnx.load("models/Llama-3-1-8B-Instruct-LoRA-Mutated/model/model.onnx", load_external_data=False)
for i in m.graph.input:
    print(i.name)