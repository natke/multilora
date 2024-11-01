import onnxruntime_genai as og
import numpy as np

model_name = "Llama-3.2-1B-Instruct-LoRA"
model = og.Model(f"models/{model_name}/adapted/model")
adapters = og.Adapters(model)
adapters.load(f"models/{model_name}/adapted/model/adapter_weights.onnx_adapter", "functions")
#adapters.unload("adapter")
# Phi-3
#prompt_template = '<|user|>\n{input} <|end|>\n<|assistant|>'
# Llama-3
prompt_template = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an in car virtual assistant that maps user\'s inputs to the corresponding function call in the vehicle. You must respond with only a JSON object matching the following schema: {{"function_name\": <name of the function>, \"arguments\": <arguments of the function>}}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
prompt = prompt_template.format(input="Please call Anthony")

tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

params = og.GeneratorParams(model)
params.set_search_options(max_length=2048, past_present_share_buffer=False)
params.input_ids = tokenizer.encode(prompt)

generator = og.Generator(model, params)

#print(f"[Base]: {prompt}")

generator.set_active_adapter(adapters, "functions")
print(f"[Adapter]: {prompt}")

while not generator.is_done():
    generator.compute_logits()
    generator.generate_next_token()

    new_token = generator.get_next_tokens()[0]
    print(tokenizer_stream.decode(new_token), end='', flush=True)

del generator


#generator = og.Generator(model, params)

#generator.set_active_adapter(adapters, "hillbilly")

#print("\n")
#print(f"[Hillbilly]: Tell me about yourself")

#while not generator.is_done():
#    generator.compute_logits()
#    generator.generate_next_token()

#    new_token = generator.get_next_tokens()[0]
#    print(tokenizer_stream.decode(new_token), end='', flush=True)

