import onnxruntime_genai as og

model = og.Model("models/Llama-3-1-8B-Instruct-LoRA/model")
adapters = og.Adapters(model)
adapters.load("adapters/Llama-1-8B-Instruct-Surfer-Dude-Personality", "surfer-dude")
adapters.load("adapters/Llama-1-8B-Instruct-Hillbilly-Personality", "hillbilly")

tokenizer = og.Tokenizer(model)
tokenizer_stream = og.TokenizerStream(model)cd ..\

params = og.GeneratorParams(model)
params.set_search_options(max_length=20)
params.input_ids = tokenizer.encode("Tell me a little about yourself")

generator = og.Generator(model, params)

generator.set_active_adapter(adapters, "surfer-dude")

while not generator.is_done():
    generator.compute_logits()
    generator.generate_next_token()

    new_token = generator.get_next_tokens()[0]
    print(tokenizer_stream.decode(new_token), end='', flush=True)

generator.set_active_adapter(adapters, "hillbilly")

