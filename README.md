# Run Multi LoRA with ONNX models

## Setup

1. Install Olive

   ```bash
   pip install git+https://github.com/microsoft/olive
   ```

2. Install ONNX Runtime nightly
   
   ```bash
   pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ ort-nightly
   ```

3. Install other dependencies

   ```bash
   pip install optimum peft
   ```

4. Build and install ONNX Runtime generate()

   

5. Choose a model

   In this example we'll use [Llama-3-8b](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

   You need to register with Meta for a license to use this model. You can do this by accessing the above page, signing in, and registering for access. Access should be granted quickly.
   
5. Locate datasets and/or existing adapters

   In this example, we will two pre-tuned adapters

   * [Coldstart/Llama-3.1-8B-Instruct-Surfer-Dude-Personality](https://huggingface.co/Coldstart/Llama-3.1-8B-Instruct-Surfer-Dude-Personality)
   * [Coldstart/Llama-3.1-8B-Instruct-Hillbilly-Personality](https://huggingface.co/Coldstart/Llama-3.1-8B-Instruct-Hillbilly-Personality)

## Generate adapters in .onnx_adapter format


### Convert an existing adapter into ONNX format

For the second adapter, we will use a pre-tuned adapter. Note the output path cannot have any period (`.`) characters.

Note also that this step requires xxx GB of memory on the machine on which it is running.

```bash
olive capture-onnx-graph -m meta-llama/Llama-3.1-8B-Instruct -a Coldstart/Llama-3.1-8B-Instruct-Surfer-Dude-Personality -o models\Llama-3-1-8B-Instruct-LoRA --dtype float32
```


```bash
olive convert-adapters --adapter_path Coldstart/Llama-3.1-8B-Instruct-Surfer-Dude-Personality --output_path adapters\Llama-1-8B-Instruct-Surfer-Dude-Personality --dtype float32
```

```bash
olive convert-adapters --adapter_path Coldstart/Llama-3.1-8B-Instruct-Hillbilly-Personality --output_path adapters\Llama-1-8B-Instruct-Hillbilly-Personality --dtype float32
```

## Write your application

```python
model = og.Model("<model/path>")
adapters = og.Adapters(model)
adapters.load("adapters\Llama-1-8B-Instruct-Surfer-Dude-Personality", "surfer-dude")
adapters.load("adapters\Llama-1-8B-Instruct-Hillbilly-Personality", "hillbilly")

tokenizer = og.Tokenizer(model)

params = og.GeneratorParams(model)
params.set_search_options(max_length=20)
params.input_ids = tokenizer.encode("Tell me a little about yourself)

generator = og.Generator(model, params)

generator.set_active_adapter(adapters, "surfer-dude")

while not generator.is_done():
    generator.compute_logits()
    generator.generate_next_token()

   new_token = generator.get_next_tokens()[0]
   print(tokenizer_stream.decode(new_token), end='', flush=True)

generator.set_active_adapter(adapters, "hillbilly")

```



## Appendix:

### Fine-tune the model with a dataset

TODO: this requires CUDA

```bash
olive finetune --method qlora -m meta-llama/Meta-Llama-3-8B -d nampdn-ai/tiny-codes --train_split "train[:4096]" --eval_split "train[4096:4224]" --text_template "### Language: {programming_language} \n### Question: {prompt} \n### Answer: {response}" --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --max_steps 150 --logging_steps 50 -o adapters\tiny-codes
```





