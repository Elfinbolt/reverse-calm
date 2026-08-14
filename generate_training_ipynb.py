import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Reverse-CALM 100-Example Training Loop\n",
    "\n",
    "This notebook trains the CALM bridge on the 100-example dataset. DeepSeek and Qwen are kept frozen, and only the bridge is trained. \n",
    "Ensure your dataset is mounted at `/kaggle/input/datasets/uddhavnaik/reverse-calm-data-100/dataset.jsonl`."
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
    "import time\n",
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
    "# Freezing\n",
    "for param in model.aug_model.parameters():\n",
    "    param.requires_grad = False\n",
    "for param in model.anchor_model.parameters():\n",
    "    param.requires_grad = False\n",
    "for param in model.cross_attention_hooks.parameters():\n",
    "    param.requires_grad = True\n",
    "\n",
    "model.aug_model.eval()\n",
    "model.anchor_model.eval()\n",
    "model.train()\n",
    "\n",
    "print(\"Trainable params:\", sum(p.numel() for p in model.parameters() if p.requires_grad))"
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
    "# 4. Training Loop\n",
    "optimizer = torch.optim.AdamW(\n",
    "    model.cross_attention_hooks.parameters(),\n",
    "    lr=1e-4,\n",
    ")\n",
    "\n",
    "num_epochs = 1\n",
    "total_examples = len(dataset)\n",
    "\n",
    "print(f\"\\nStarting Training for {num_epochs} epoch(s)...\")\n",
    "start_time = time.time()\n",
    "\n",
    "for epoch in range(num_epochs):\n",
    "    epoch_loss = 0.0\n",
    "    \n",
    "    for i, example in enumerate(dataset):\n",
    "        # Prepare Text\n",
    "        reasoning_text = (\n",
    "            f\"Question: {example['question_konkani']}\\n\\n\"\n",
    "            f\"Reasoning: {example['reasoning_english']}\"\n",
    "        )\n",
    "        target_text = example[\"reasoning_konkani\"]\n",
    "\n",
    "        # Tokenize\n",
    "        aug_inputs = aug_tokenizer(\n",
    "            reasoning_text,\n",
    "            return_tensors=\"pt\",\n",
    "            truncation=True,\n",
    "            max_length=512,\n",
    "        )\n",
    "        \n",
    "        target_inputs = anchor_tokenizer(\n",
    "            target_text,\n",
    "            return_tensors=\"pt\",\n",
    "            truncation=True,\n",
    "            max_length=128,\n",
    "        )\n",
    "\n",
    "        # Move to device\n",
    "        aug_inputs = {k: v.to(\"cpu\") for k, v in aug_inputs.items()}\n",
    "        target_inputs = {k: v.to(device) for k, v in target_inputs.items()}\n",
    "\n",
    "        # Teacher Forcing\n",
    "        qwen_input_ids = target_inputs[\"input_ids\"][:, :-1]\n",
    "        qwen_attention_mask = target_inputs[\"attention_mask\"][:, :-1]\n",
    "        qwen_labels = target_inputs[\"input_ids\"][:, 1:].clone()\n",
    "\n",
    "        # Forward Pass\n",
    "        optimizer.zero_grad()\n",
    "        outputs = model(\n",
    "            input_ids=qwen_input_ids,\n",
    "            attention_mask=qwen_attention_mask,\n",
    "            aug_input_ids=aug_inputs[\"input_ids\"],\n",
    "            aug_attention_mask=aug_inputs[\"attention_mask\"],\n",
    "            labels=qwen_labels,\n",
    "            use_cache=False,\n",
    "        )\n",
    "        \n",
    "        loss = outputs.loss\n",
    "        \n",
    "        # Backward Pass & Optimize\n",
    "        loss.backward()\n",
    "        # torch.nn.utils.clip_grad_norm_(model.cross_attention_hooks.parameters(), max_norm=1.0) # Optional clipping\n",
    "        optimizer.step()\n",
    "        \n",
    "        epoch_loss += loss.item()\n",
    "        \n",
    "        # Logging\n",
    "        if (i + 1) % 5 == 0 or (i + 1) == total_examples:\n",
    "            print(f\"Epoch {epoch+1}/{num_epochs} | Example {i+1:3d}/{total_examples} | Loss: {loss.item():.4f}\")\n",
    "\n",
    "    avg_loss = epoch_loss / total_examples\n",
    "    print(f\"\\n--- Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f} ---\\n\")\n",
    "\n",
    "total_time = time.time() - start_time\n",
    "print(f\"Training finished in {total_time:.2f} seconds.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 5. Save the Checkpoint\n",
    "checkpoint_path = \"/kaggle/working/reverse_calm_bridge.pt\"\n",
    "torch.save(model.cross_attention_hooks.state_dict(), checkpoint_path)\n",
    "print(f\"Bridge checkpoint saved to {checkpoint_path}\")"
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
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open(os.path.join("c:\\Data\\csrc\\reverse-calm", "Kaggle_Training_Loop.ipynb"), "w") as f:
    json.dump(notebook, f, indent=2)
print("Training Notebook generated.")
