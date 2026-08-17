import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Reverse-CALM Inference Test\n",
    "\n",
    "This notebook runs the trained CALM bridge to generate a Konkani response.\n",
    "Ensure your dataset is mounted at `/kaggle/input/datasets/uddhavnaik/reverse-calm-data-100/dataset_v2.jsonl` and that you have `reverse_calm_bridge.pt` in the working directory."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import torch\n",
    "import json\n",
    "from transformers import AutoTokenizer\n",
    "\n",
    "# Ensure Kaggle environment can find our repo\n",
    "REPO_PATH = \"/kaggle/working/reverse-calm\"\n",
    "if REPO_PATH not in sys.path:\n",
    "    sys.path.insert(0, REPO_PATH)\n",
    "\n",
    "from model.calm import CALM, CALMConfig"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Load Dataset\n",
    "DATA_PATH = \"/kaggle/input/datasets/uddhavnaik/reverse-calm-data-100/dataset_v2.jsonl\"\n",
    "with open(DATA_PATH, \"r\", encoding=\"utf-8\") as f:\n",
    "    dataset = [json.loads(line) for line in f]\n",
    "\n",
    "print(\"Total Examples:\", len(dataset))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. Initialize Model\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(f\"Using device: {device}\")\n",
    "\n",
    "config = CALMConfig(\n",
    "    anchor_model=\"nischay185/konkani-qwen2-1.5b\",\n",
    "    aug_model=\"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B\",\n",
    "    num_connections=4,\n",
    "    num_heads=4,\n",
    ")\n",
    "\n",
    "model = CALM(config)\n",
    "\n",
    "# Model Placement\n",
    "model.aug_model = model.aug_model.to(\"cpu\")\n",
    "model.anchor_model = model.anchor_model.to(device)\n",
    "model.cross_attention_hooks = model.cross_attention_hooks.to(device)\n",
    "model.cross_attention_hooks.float() # FP32 for numerical stability\n",
    "\n",
    "# Load Trained Bridge Weights\n",
    "weights_path = \"/kaggle/working/reverse_calm_bridge.pt\"\n",
    "if os.path.exists(weights_path):\n",
    "    model.cross_attention_hooks.load_state_dict(torch.load(weights_path))\n",
    "    print(\"Successfully loaded trained bridge weights!\")\n",
    "else:\n",
    "    print(\"WARNING: Trained weights not found. Running with untrained bridge.\")\n",
    "\n",
    "model.aug_model.eval()\n",
    "model.anchor_model.eval()\n",
    "model.eval()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. Tokenizers\n",
    "anchor_tokenizer = AutoTokenizer.from_pretrained(config.anchor_model)\n",
    "aug_tokenizer = AutoTokenizer.from_pretrained(config.aug_model)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. Generation Test\n",
    "example_idx = 0\n",
    "example = dataset[example_idx]\n",
    "\n",
    "reasoning_text = (\n",
    "    f\"Question: {example['question_konkani']}\\n\\n\"\n",
    "    f\"Reasoning: {example['reasoning_english']}\"\n",
    ")\n",
    "\n",
    "aug_inputs = aug_tokenizer(\n",
    "    reasoning_text,\n",
    "    return_tensors=\"pt\",\n",
    "    truncation=True,\n",
    "    max_length=512,\n",
    ").to(\"cpu\")\n",
    "\n",
    "# Provide Qwen with a BOS token to kick off generation\n",
    "input_ids = torch.tensor([[anchor_tokenizer.bos_token_id]]).to(device)\n",
    "\n",
    "print(\"Generating...\\n\")\n",
    "with torch.no_grad():\n",
    "    outputs = model.generate(\n",
    "        input_ids=input_ids,\n",
    "        aug_input_ids=aug_inputs.input_ids,\n",
    "        aug_attention_mask=aug_inputs.attention_mask,\n",
    "        max_new_tokens=100,\n",
    "        pad_token_id=anchor_tokenizer.eos_token_id,\n",
    "        do_sample=False,\n",
    "    )\n",
    "\n",
    "output_text = anchor_tokenizer.decode(outputs[0], skip_special_tokens=True)\n",
    "\n",
    "print(\"--- INPUT TO BRIDGE (DeepSeek) ---\")\n",
    "print(reasoning_text)\\n",
    "\n",
    "print(\"\\n--- GENERATED KONKANI (Qwen) ---\")\n",
    "print(output_text)\\n",
    "\n",
    "print(\"\\n--- EXPECTED REASONING ---\")\n",
    "print(example['reasoning_konkani'])\\n",
    "\n",
    "print(\"\\n--- EXPECTED FINAL ANSWER ---\")\n",
    "print(example['answer_konkani'])\\n",
    "model.release_memory()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("Kaggle_Inference.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Inference Notebook generated.")
