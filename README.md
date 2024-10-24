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

5. Downgrade torch

   TODO: There is an export bug with torch 2.5.0

   ```bash
   pip uninstall torch
   pip install torch==2.4
   ```
   
6. Choose a model

   In this example we'll use [Llama-3-8b](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

   You need to register with Meta for a license to use this model. You can do this by accessing the above page, signing in, and registering for access. Access should be granted quickly. Esnure that the huggingface-cli is installed (`pip install huggingface-hub[cli]`) and you are logged in via `huggingface-cli login`.
   
7. Locate datasets and/or existing adapters

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
   olive quantize -m Llama-3-1-8B-Instruct-LoRA\model --algorithm rtn --implementation matmul4 -o Llama-3-1-8B-Instruct-LoRA-int4
   ```

3. Adapt model

   ```bash
   olive generate-adapter -m models\Llama-3-1-8B-Instruct-LoRA-int4\model -o models\Llama-3-1-8B-Instruct-LoRA-int4\adapted -log_level 1
   ```

4. Convert adapters to ONNX

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





