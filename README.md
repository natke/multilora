# Run Multi LoRA with ONNX models

## Setup

1. Install Olive

   ```bash
   pip install git+https://github.com/microsoft/olive
   ```

2. Build and install ONNX Runtime generate()

   TODO: replace this with 1.20 when it is released

   ```bash
   git clone https://github.com/microsoft/onnxruntime-genai.git
   cd onnxruntime-genai
   python build.py
   cd build\Windows\RelWithDebInfo\wheel
   pip install *.whl

3. Install ONNX Runtime nightly
   
   TODO: remove this step when 1.20 is released

   ```bash
   pip uninstall onnxruntime
   pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ ort-nightly
   ```

4. Install other dependencies

   ```bash
   pip install optimum peft
   ```

5. Choose a model

   In this example we'll use [Llama-3-8b](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

   You need to register with Meta for a license to use this model. You can do this by accessing the above page, signing in, and registering for access. Access should be granted quickly.
   
5. Locate datasets and/or existing adapters

   In this example, we will two pre-tuned adapters

   * [Coldstart/Llama-3.1-8B-Instruct-Surfer-Dude-Personality](https://huggingface.co/Coldstart/Llama-3.1-8B-Instruct-Surfer-Dude-Personality)
   * [Coldstart/Llama-3.1-8B-Instruct-Hillbilly-Personality](https://huggingface.co/Coldstart/Llama-3.1-8B-Instruct-Hillbilly-Personality)

## Generate model and adapters in ONNX format

### Convert existing adapters into ONNX format

Note the output path cannot have any period (`.`) characters.

Note also that this step requires 37GB of memory on the machine on which it is running.

1. Export the model to ONNX format

   Note: add --use_model_builder when this is ready

   ```bash
   olive capture-onnx-graph -m meta-llama/Llama-3.1-8B-Instruct -a Coldstart/Llama-3.1-8B-Instruct-Surfer-Dude-Personality -o models\Llama-3-1-8B-Instruct-LoRA --dtype float32
   python -m onnxruntime_genai.models.builder meta-llama/Llama-3.1-8B-Instruct -e cpu -p fp32 --extra_options config_only=true
   ```

2. Mutate model

   ```bash
   olive generate-adapter -m models\Llama-3-1-8B-Instruct-LoRA\model -o models\Llama-3-1-8B-Instruct-LoRA\mutated -log_level 1
   ```

3. Convert adapters to ONNX

   ```bash
   olive convert-adapters --adapter_path Coldstart/Llama-3.1-8B-Instruct-Surfer-Dude-Personality --output_path adapters\Llama-1-8B-Instruct-Surfer-Dude-Personality --dtype float32
   ```

   ```bash
   olive convert-adapters --adapter_path Coldstart/Llama-3.1-8B-Instruct-Hillbilly-Personality --output_path adapters\Llama-1-8B-Instruct-Hillbilly-Personality --dtype float32
   ```

## Write your application

See [app.py](app.py)


## Appendix:

### Fine-tune the model with a dataset

TODO: this requires CUDA

```bash
olive finetune --method qlora -m meta-llama/Meta-Llama-3-8B -d nampdn-ai/tiny-codes --train_split "train[:4096]" --eval_split "train[4096:4224]" --text_template "### Language: {programming_language} \n### Question: {prompt} \n### Answer: {response}" --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --max_steps 150 --logging_steps 50 -o adapters\tiny-codes
```





