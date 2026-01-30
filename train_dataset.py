import json
from random import randint
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from load_model import tokenizer

# Load dataset
data_file = ""
dataset = load_dataset("json", data_files=data_file)["train"].shuffle()

print(f"\n\033[Dataset Sample\033[0m \n{json.dumps(dataset[randint(0, len(dataset) - 1)], indent=2)}")


# Process dataset
def apply_format(sample):
  template_iputs = sample

  prompt_and_completion = tokenizer.apply_chat_template(
    template_iputs['messages'],
    tools=template_iputs['tools'],
    tokenize=False,
    # add_generation_prompt is False since we don't need model output after all
    # messages.
    add_generation_prompt=False)

  prompt = tokenizer.apply_chat_template(
    template_iputs['messages'][:-1],
    tools=template_iputs['tools'],
    tokenize=False,
    # add_generation_prompt is True since we would like to include
    # "<start_of_turn>model" in the prompt, if needed.
    add_generation_prompt=True)

  completion = prompt_and_completion[len(prompt):]

  return {
     "prompt": prompt,
     "completion": completion,
     "split": template_iputs["metadata"],
  }

processed_dataset = dataset.map(apply_format)
