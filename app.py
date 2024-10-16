import onnxruntime_genai as og
import numpy as np

model = og.Model("models/Llama-3-1-8B-Instruct-LoRA/mutated/model")
adapters = og.Adapters(model)
adapters.load("adapters/Llama-1-8B-Instruct-Surfer-Dude-Personality.onnx_adapter", "surfer-dude")
adapters.load("adapters/Llama-1-8B-Instruct-Hillbilly-Personality.onnx_adapter", "hillbilly")

tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

params = og.GeneratorParams(model)
params.set_search_options(max_length=100, past_present_share_buffer=False)
params.input_ids = tokenizer.encode("Tell me a little about yourself")
params.set_model_input("onnx::Neg_67", np.array(0, dtype=np.int64))

generator = og.Generator(model, params)

generator.set_active_adapter(adapters, "surfer-dude")

while not generator.is_done():
    generator.compute_logits()
    generator.generate_next_token()

    new_token = generator.get_next_tokens()[0]
    print(tokenizer_stream.decode(new_token), end='', flush=True)

del generator


generator = og.Generator(model, params)

generator.set_active_adapter(adapters, "hillbilly")

while not generator.is_done():
    generator.compute_logits()
    generator.generate_next_token()

    new_token = generator.get_next_tokens()[0]
    print(tokenizer_stream.decode(new_token), end='', flush=True)

