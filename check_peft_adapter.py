from transformers import AutoModelForCausalLM
from peft import PeftModel 
import argparse

parser = argparse.ArgumentParser(description='Check model inputs')
parser.add_argument('model', type=str, help='Model path')
parser.add_argument('adapter', type=str, help='Adapter path')
parser.add_argument('tensor', type=str, help='Tensor name')

args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model)
adapter = PeftModel.from_pretrained(model, args.adapter, use_safetensors=True)

print(adapter.state_dict()[args.tensor].shape)  