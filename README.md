# Run Multi LoRA with ONNX models

## Setup

1. Install Olive
   
   This installs Olive from main. Replace with version 0.8.0 when it is released.

   ```bash
   pip install git+https://github.com/microsoft/olive
   ```

2. Install ONNX Runtime generate()

   ```
   pip install onnxruntime-genai
   ```

3. Install other dependencies

   ```bash
   pip install optimum peft
   ```

4. Downgrade torch and transformers

   TODO: There is an export bug with torch 2.5.0 and an incompatibility with transformers>=4.45.0

   ```bash
   pip uninstall torch
   pip install torch==2.4
   pip uninstall transformers
   pip install transformers==4.44
   ```
   
5. Choose a model

   In this example we'll use [Llama-3-8b](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

   You need to register with Meta for a license to use this model. You can do this by accessing the above page, signing in, and registering for access. Access should be granted quickly. Esnure that the huggingface-cli is installed (`pip install huggingface-hub[cli]`) and you are logged in via `huggingface-cli login`.
   
6. Locate datasets and/or existing adapters

   In this example, we will two pre-tuned adapters

   * [Coldstart/Llama-3.1-8B-Instruct-Surfer-Dude-Personality](https://huggingface.co/Coldstart/Llama-3.1-8B-Instruct-Surfer-Dude-Personality)
   * [Coldstart/Llama-3.1-8B-Instruct-Hillbilly-Personality](https://huggingface.co/Coldstart/Llama-3.1-8B-Instruct-Hillbilly-Personality)

## Generate model and adapters in ONNX format

### Convert existing adapters into ONNX format

Note the output path cannot have any period (`.`) characters.

Note also that this step requires 63GB of memory on the machine on which it is running.

1. Export the model to ONNX format

   Note: add --use_model_builder when this is ready

   ```bash
   olive capture-onnx-graph -m meta-llama/Llama-3.1-8B-Instruct --adapter_path Coldstart/Llama-3.1-8B-Instruct-Surfer-Dude-Personality -o models\Llama-3-1-8B-Instruct-LoRA --torch_dtype float32 --use_ort_genai
   ```

2. (Optional) Quantize the model

   ```bash
   olive quantize -m models\Llama-3-1-8B-Instruct-LoRA --algorithm rtn --implementation matmul4 -o models\Llama-3-1-8B-Instruct-LoRA-int4
   ```

3. Adapt model

   ```bash
   olive generate-adapter -m models\Llama-3-1-8B-Instruct-LoRA-int4 -o models\Llama-3-1-8B-Instruct-LoRA-int4\adapted
   ```

4. Convert adapters to ONNX

   This steps assumes you quantized the model in Step 2. If you skipped step 2, then remove the `--quantize_int4` argument.

   ```bash
   olive convert-adapters --adapter_path Coldstart/Llama-3.1-8B-Instruct-Surfer-Dude-Personality --output_path adapters\Llama-3.1-8B-Instruct-Surfer-Dude-Personality --dtype float32 --quantize_int4
   ```

   ```bash
   olive convert-adapters --adapter_path Coldstart/Llama-3.1-8B-Instruct-Hillbilly-Personality --output_path adapters\Llama-3.1-8B-Instruct-Hillbilly-Personality --dtype float32 --quantize_int4
   ```

## Write your application

See [app.py](app.py) as an example.

## Call the application

```bash
python app.py -m models\Llama-3-1-8B-Instruct-LoRA-int4\adapted\model -a adapters\adapters\Llama-3.1-8B-Instruct-Hillbilly-Personality.onnx_adapter -t "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" -s "You are a friendly chatbot" -p "Hi, how are you today?"
```


## Appendix:

### Fine-tune your own data

Note: this requires CUDA

Use the `olive fine-tune` command: https://microsoft.github.io/Olive/features/cli.html#finetune

Here is an example usage of the commmand:

```bash
olive finetune --method qlora -m meta-llama/Meta-Llama-3-8B -d nampdn-ai/tiny-codes --train_split "train[:4096]" --eval_split "train[4096:4224]" --text_template "### Language: {programming_language} \n### Question: {prompt} \n### Answer: {response}" --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --max_steps 150 --logging_steps 50 -o adapters\tiny-codes
```





