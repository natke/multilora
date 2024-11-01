import onnxruntime_genai as og
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Application to load and switch ONNX LoRA adapters')
parser.add_argument('-m', '--model', type=str, help='The ONNX base model')
parser.add_argument('-a', '--adapters', nargs='+', type=str, help='List of adapters in .onnx_adapters format')
parser.add_argument('-t', '--template', type=str, help='The template with which to format the prompt')
parser.add_argument('-s', '--system', type=str, help='The system prompt to pass to the model')
parser.add_argument('-p', '--prompt', type=str, help='The user prompt to pass to the model')
args = parser.parse_args()

model = og.Model(args.model)
adapters = og.Adapters(model)
for adapter in args.adapters:
    adapters.load(adapter, adapter)

tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

# Example prompt templates
# Phi-3 
#   <|user|>\n{input} <|end|>\n<|assistant|>
# Llama-2
#  "<s>{input}"
#  "<s>[INST] <<SYS>>\nAnswer as briefly as possible\n<</SYS>>\n\n{input}  [/INST]"
#  "System: You are a helpful and honest assistant\nPrompt: {input}\nResponse:\n"
# Llama-3
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n

prompt = args.template.format(system=args.system, input=args.prompt)

params = og.GeneratorParams(model)
params.set_search_options(max_length=2048, past_present_share_buffer=False)
# This input is generated for transformers versions > 4.45
#params.set_model_input("onnx::Neg_67", np.array(0, dtype=np.int64))
params.input_ids = tokenizer.encode(prompt)

generator = og.Generator(model, params)

for adapter in args.adapters:
   print(f"[{adapter}]: {prompt}")
   generator.set_active_adapter(adapters, adapter)

while not generator.is_done():
    generator.compute_logits()
    generator.generate_next_token()

    new_token = generator.get_next_tokens()[0]
    print(tokenizer_stream.decode(new_token), end='', flush=True)
