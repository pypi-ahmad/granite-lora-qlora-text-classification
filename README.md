# LoRA and QLoRA Fine-Tuning on Granite 4.1 3B — AG News Text Classification

End-to-end tutorial demonstrating **LoRA** and **QLoRA** fine-tuning of IBM Granite 4.1 3B on a 4-class text classification task (AG News), running entirely on a single consumer GPU with 8 GB VRAM.

This notebook covers everything from dataset preparation and zero-shot baseline evaluation through adapter training, quantized training, and a final three-way accuracy comparison — with detailed explanations of every memory-optimization technique required to fit a 3-billion-parameter model onto 8 GB of VRAM.

---

## Overview

**What it does:**  
Fine-tunes `ibm-granite/granite-4.1-3b-base` for news-category classification using two PEFT strategies:
- **LoRA** — trainable low-rank adapter matrices injected into frozen base model weights
- **QLoRA** — same LoRA adapters on top of a 4-bit NF4 quantized base, reducing base-model VRAM from ~6 GB to ~1.7 GB

**Why it exists:**  
Large language model fine-tuning on consumer hardware has a steep learning curve — small misconfigurations (wrong `dtype` key, un-freed VRAM references, incompatible TrainingArguments) cause silent failures or OOM crashes. This project documents a fully working, debugged pipeline with every gotcha explained.

**Key objectives:**
1. Demonstrate end-to-end PEFT on a 3B model using only 8 GB VRAM
2. Quantify the accuracy and VRAM trade-off between LoRA, QLoRA, and the zero-shot baseline
3. Provide a production-quality template for fine-tuning decoder-only LLMs for classification

---

## Features

- Zero-shot baseline evaluation using generation-based classification
- LoRA fine-tuning with r=16 on all linear layers (`all-linear` target)
- QLoRA fine-tuning with 4-bit NF4 quantization (double quantization enabled)
- Gradient checkpointing, bf16 mixed precision, and paged AdamW — all active simultaneously
- Label-masking (`-100`) so cross-entropy loss is computed only over the output label
- Adapter save/reload with `PeftModel.from_pretrained` for deployment
- GPU memory profiling across all pipeline stages
- Three-way accuracy comparison table (Base vs. LoRA vs. QLoRA)
- Inference examples showing all three models on the same articles
- Comprehensive troubleshooting guide for transformers 5.x and bitsandbytes on CUDA 13

---

## Architecture

### Base Model
- **Model**: `ibm-granite/granite-4.1-3b-base`
- **Type**: Decoder-only transformer (`GraniteForCausalLM`), LLaMA-family architecture
- **Size**: 3.4 billion parameters, 32 transformer layers, hidden size 3072
- **Context length**: 128K tokens, vocabulary size 49,152
- **Disk footprint**: ~6.4 GB (bf16 safetensors shards)
- **Inference VRAM**: ~6 GB (bf16) | ~1.7 GB (4-bit NF4)

### LoRA Architecture
LoRA (Low-Rank Adaptation) injects a pair of small matrices into each frozen linear layer:

```
output = W₀x + (B · A)x · (α / r)
```

Where `W₀` is frozen, `A ∈ ℝ^(r×d_in)` and `B ∈ ℝ^(d_out×r)` are the trainable adapter matrices, `r=16` is the rank, and `α=32` is the scaling factor.

| Parameter | Value |
|-----------|-------|
| Rank `r` | 16 |
| `lora_alpha` | 32 |
| Effective LR multiplier | α/r = 2.0 |
| `lora_dropout` | 0.05 |
| `target_modules` | `all-linear` (every `nn.Linear`) |
| Trainable parameters | ~31 M / 3.4 B = **0.9%** |
| Adapter checkpoint size | ~62 MB |

### QLoRA Architecture
QLoRA adds NF4 quantization of the frozen base model on top of LoRA:

```
W₀ (frozen, bf16) → W₀_quantized (4-bit NF4)
Adapter matrices A, B remain in bf16
```

| Component | LoRA | QLoRA |
|-----------|------|-------|
| Base model dtype | bf16 | 4-bit NF4 |
| Base model VRAM | ~6 GB | ~1.7 GB |
| Adapter dtype | bf16 | bf16 |
| Compute dtype | bf16 | bf16 |
| Double quantization | — | ✓ (quant constants compressed) |
| Total training VRAM | ~7.7 GB | ~4.5–5.5 GB |

### Quantization Approach (NF4)
Normal Float 4 (NF4) assigns 16 quantization bins optimally positioned for normally-distributed weights. Advantages over standard INT4:
- Quantization error is minimized for zero-mean Gaussian weight distributions
- Double quantization compresses the per-block quantization constants from FP32 to FP8, saving an additional ~0.5 GB
- `bnb_4bit_compute_dtype=bfloat16` dequantizes on-the-fly to bf16 for each matrix multiply

### Training Pipeline
```
Dataset → Tokenize (masked labels) → Trainer.train()
  ├── Forward pass (frozen base + active adapters)
  ├── Loss on label tokens only (prompt masked with -100)
  ├── Backward pass through adapters only (base has requires_grad=False)
  ├── Gradient accumulation × 16 steps
  └── AdamW optimizer step on adapter parameters
```

### Evaluation Pipeline
Two evaluation modes are used:
1. **Token-level loss** (`Trainer.evaluate()`): cross-entropy on validation set, fast
2. **Generation-based accuracy** (`evaluate_generation_classifier()`): greedy decode → parse → compare to ground truth label, applied to 2000 test samples

### Inference Pipeline
```
Article text → Prompt template → Tokenize → model.generate() [greedy, max_new_tokens=8]
  → Decode output → Strip prompt → Parse label token → Map to class ID
```

---

## Dataset

**Name**: [AG News](https://huggingface.co/datasets/fancyzhx/ag_news)  
**Task**: 4-class news topic classification  
**Source**: HuggingFace Datasets (`fancyzhx/ag_news`)

### Classes
| ID | Label | Description |
|----|-------|-------------|
| 0 | World | International news and current events |
| 1 | Sports | Sports news and results |
| 2 | Business | Business, finance, and economics |
| 3 | Sci/Tech | Science and technology news |

### Splits Used
| Split | Samples | Purpose |
|-------|---------|---------|
| Train | 110,000 | Fine-tuning adapters |
| Validation | 10,000 | Epoch-end evaluation (token-level loss) |
| Generation eval | 2,000 | Generation-based accuracy measurement |

Classes are balanced at ~25% each.

### Preprocessing
1. Load with `datasets.load_dataset("fancyzhx/ag_news")`
2. Split original 120K train into 110K train / 10K validation (stratified shuffle)
3. Wrap each sample in a prompt template:
   ```
   Classify the following news article into one of: World, Sports, Business, Sci/Tech.
   Return only the label.
   Article: {text}
   Label:
   ```
4. Tokenize with `AutoTokenizer`, padding to `MAX_SEQ_LENGTH=384`
5. Set prompt token labels to `-100` so only the label token contributes to loss
6. Tokenizer's pad token set to `eos_token` for causal LM compatibility

---

## Hardware Requirements

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM | 8,188 MiB (8 GB) |
| RAM | 16 GB+ recommended |
| Storage | ~15 GB (model cache ~6.4 GB + venv ~5.6 GB + data ~1 GB) |
| CUDA Compute | 8.9 (Ada Lovelace) |

**Minimum requirements:**
- LoRA training: ~7.7 GB VRAM (use `MAX_SEQ_LENGTH=256` if you have exactly 8 GB and hit OOM)
- QLoRA training: ~4.5–5.5 GB VRAM (runs comfortably on 6 GB cards)
- CPU inference: supported but slow (model loads on CPU if no GPU detected)

---

## Software Requirements

| Component | Version |
|-----------|---------|
| OS | Ubuntu Linux 7.0.0 |
| Python | 3.12.10 |
| CUDA Toolkit | 13.1 |
| NVIDIA Driver | 595.71.05 |
| PyTorch | 2.12.0+cu130 |
| Transformers | 5.12.0 |
| PEFT | 0.19.1 |
| Datasets | 5.0.0 |
| Accelerate | 1.14.0 |
| BitsAndBytes | 0.49.2 |
| HuggingFace Hub | 1.19.0 |
| uv (package manager) | 0.11.19 |

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/pypi-ahmad/granite-lora-qlora-text-classification.git
cd granite-lora-qlora-text-classification
```

### 2. Install uv (recommended)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Create environment and install dependencies
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

For CUDA 12.x (adjust index URL to match your CUDA version):
```bash
uv pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt --no-deps
uv pip install -r requirements.txt
```

### 4. Install Jupyter kernel
```bash
.venv/bin/python -m ipykernel install --user --name granite-lora-env --display-name "Granite LoRA/QLoRA"
```

### 5. Verify GPU access
```bash
.venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0)); print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
```

---

## Environment Setup

Set the memory allocator variable before any CUDA operations to reduce fragmentation:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This is also set programmatically inside the notebook at import time.

To authenticate with HuggingFace (required for gated models, not needed for Granite):
```bash
huggingface-cli login
```

---

## Project Structure

```
granite-lora-qlora-text-classification/
├── granite41_lora_qlora_text_finetuning.ipynb   # Main tutorial notebook (39 cells)
├── requirements.txt                              # Pinned Python dependencies
├── README.md                                     # This file
├── .gitignore
├── scripts/
│   ├── run_granite41_gpu_stages.sh              # Stage-by-stage nbconvert runner with OOM retry
│   └── gpu_guard.sh                             # VRAM guard: kills stale GPU procs, waits for free VRAM
└── artifacts/
    ├── lora_adapter/          # Full-run LoRA adapter configs + tokenizer (weights excluded, see below)
    ├── demo_lora_adapter/     # 20K-sample demo LoRA adapter configs + tokenizer
    └── demo_qlora_adapter/    # 20K-sample demo QLoRA adapter configs + tokenizer
```

**What's excluded** (see `.gitignore`):
- `*.safetensors` — adapter weight files are 119 MB each, exceeding GitHub's 100 MB limit; run the notebook to reproduce them
- `.venv/` — 5.6 GB virtual environment (recreate with `uv pip install -r requirements.txt`)
- `artifacts/lora_run/`, `artifacts/demo_lora_run/`, `artifacts/demo_qlora_run/` — HF Trainer checkpoint directories
- `.ipynb_checkpoints/` — Jupyter auto-save files

---

## Training

### LoRA Fine-Tuning

```python
from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",  # applies to every nn.Linear
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(base_model, config)
```

With `r=16` on `all-linear`, the adapter adds ~31M parameters (0.9% of the 3.4B base).

### QLoRA Fine-Tuning

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map={"": 0}
)
base_model = prepare_model_for_kbit_training(base_model)
qlora_model = get_peft_model(base_model, config)  # same LoRA config
```

### Hyperparameters

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| `num_train_epochs` | 1 | Single pass over 110K samples |
| `learning_rate` | 2e-4 | Higher than full fine-tuning; adapters start from zero |
| `per_device_train_batch_size` | 1 | Tight VRAM — 1 sample at a time |
| `gradient_accumulation_steps` | 16 | Effective batch = 1×16 = 16 |
| `warmup_ratio` | 0.03 | First 3% of steps for adapter stabilization |
| `weight_decay` | 0.01 | Mild L2 regularization |
| `max_seq_length` | 384 | Covers 99%+ of AG News prompts |
| `optim` | `adamw_torch` | Standard AdamW (use `paged_adamw_8bit` for lower VRAM) |

### Memory Optimization Techniques

All five of these are required simultaneously to fit a 3B model in 8 GB VRAM:

| Technique | Savings | How |
|-----------|---------|-----|
| LoRA (frozen base) | ~90% gradient/optimizer VRAM | Only 31M adapter params have gradients |
| Gradient checkpointing | ~40% activation VRAM | Recomputes activations instead of storing them |
| BF16 mixed precision | ~50% weight memory vs FP32 | All compute in half-precision |
| Gradient accumulation | ~94% batch-activation VRAM | Micro-batch=1, accumulate 16 steps |
| `use_cache=False` | ~30 MB per layer | Disables KV cache (incompatible with grad checkpointing) |

Without these, training a 3B model would require ~28 GB VRAM.

---

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| `accuracy_strict` | Exact match: predicted label == ground truth |
| `macro_f1_known_only` | Macro-averaged F1 across known classes (excludes unparseable outputs) |
| `label_parse_coverage` | Fraction of outputs that produced a valid label (quality check) |

### Validation Approach
Generation-based classification: the model generates up to 8 new tokens, the output is decoded and the first recognized class label is extracted. This mirrors real deployment — we test the actual generation capability, not just the loss.

### Results

| Model | Accuracy | Macro F1 | Parse Coverage | Peak VRAM |
|-------|----------|----------|----------------|-----------|
| Base (zero-shot) | **89.15%** | 88.96% | 100% | ~6.2 GB |
| LoRA fine-tuned (110K samples) | **95.15%** | 94.99% | 100% | ~7.6 GB |
| QLoRA fine-tuned (20K demo) | **94.2%** | 93.82% | 100% | ~4.5 GB |

> Full QLoRA on 110K samples was not completed (runtime ~16–17 h); the demo run on 20K samples confirms near-identical accuracy to LoRA at 40% lower VRAM.

**Baseline interpretation**: Granite 4.1 3B achieves 89.15% zero-shot on AG News — strong prior for a balanced 4-class task. Fine-tuning with only 20K samples (18% of the full dataset) already pushes both LoRA and QLoRA to ~94%, demonstrating rapid saturation on this balanced classification task.

---

## Inference

### Load and run a fine-tuned LoRA model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "ibm-granite/granite-4.1-3b-base"
LORA_ADAPTER_DIR = "artifacts/lora_adapter"

tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_DIR)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
model.eval()

PROMPT_TEMPLATE = (
    "Classify the following news article into one of: World, Sports, Business, Sci/Tech.\n"
    "Return only the label.\n"
    "Article: {text}\n"
    "Label:"
)

def classify(text: str) -> str:
    prompt = PROMPT_TEMPLATE.format(text=text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded.strip()

article = "NASA's Artemis program completed its first crewed lunar flyby last week."
print(classify(article))  # → Sci/Tech
```

### Load a QLoRA model (4-bit base)

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
)
qlora_model = PeftModel.from_pretrained(base_model, "artifacts/qlora_adapter")
qlora_model.eval()
```

---

## Results

### Training Summary

| Stage | Training Samples | Optimizer Steps | Duration | Final Train Loss |
|-------|-----------------|-----------------|----------|-----------------|
| LoRA (full run) | 110,000 | 6,875 | 11h 36m | 0.0737 |
| QLoRA (full run) | 110,000 | 6,875 | ~16–17 h (not completed) | — |
| LoRA (20K demo) | 20,000 | 1,250 | 2h 16m | 0.0985 |
| QLoRA (20K demo) | 20,000 | 1,250 | 3h 6m | 0.0996 |

> Demo runs use 18% of the training data. QLoRA trains 37% slower than LoRA on RTX 4060 due to NF4 dequantization overhead at batch=1; this overhead is outweighed by the ~3.2 GB VRAM savings on GPUs with ≤6 GB VRAM.

### Evaluation Summary

| Model | Accuracy | Macro F1 | Parse Coverage | Peak VRAM |
|-------|----------|----------|----------------|-----------|
| Base (zero-shot) | **89.15%** | 88.96% | 100.0% | ~6.2 GB |
| LoRA fine-tuned (110K) | **95.15%** | 94.99% | 100.0% | ~7.6 GB |
| LoRA fine-tuned (20K demo) | **94.20%** | 93.83% | 100.0% | ~7.6 GB |
| QLoRA fine-tuned (20K demo) | **94.20%** | 93.82% | 100.0% | ~4.5 GB |

### Memory Profile

| Stage | VRAM Used | Peak VRAM |
|-------|-----------|-----------|
| Base model inference | ~6.2 GB | ~6.5 GB |
| LoRA training | 7.56 GB | 7.56 GB |
| QLoRA training | 4.26 GB | 4.62 GB |
| LoRA inference | ~6.3 GB | ~6.3 GB |
| QLoRA inference | ~2.0 GB | ~2.2 GB |

---

## Memory Optimization Techniques

### 1. PEFT / LoRA
Only 0.9% of parameters (31M / 3.4B) are trainable. The remaining 99.1% of the base model has `requires_grad=False`, which eliminates:
- Gradients for frozen layers (~12 GB in fp32 or ~6 GB in bf16)
- Optimizer states (Adam maintains 2 momentum terms per trainable parameter)

### 2. QLoRA / NF4 Quantization
The frozen base model is compressed from bf16 (~6 GB) to 4-bit NF4 (~1.7 GB). Each weight is mapped to the nearest of 16 optimal NF4 bins, which are positioned to minimize quantization error for normally-distributed weights.

Double quantization additionally compresses the quantization constants: instead of one FP32 constant per 64-weight block, it stores FP8 constants per 256-weight super-block, saving ~0.5 GB.

### 3. Gradient Checkpointing
Instead of storing all intermediate activations during the forward pass (~4–8 GB for 32 layers at seq=384), gradient checkpointing discards them and recomputes from the nearest checkpoint during the backward pass. This trades ~30–40% extra compute for ~40% VRAM reduction.

```python
model.gradient_checkpointing_enable()
model.config.use_cache = False  # KV cache is incompatible with grad checkpointing
```

### 4. BF16 Mixed Precision
All weights, activations, and gradients are stored in bfloat16 (2 bytes vs. 4 for FP32), halving memory with minimal precision loss. BF16 is preferred over FP16 on Ampere+ because it has the same exponent range as FP32 (avoids overflow on gradients).

### 5. VRAM Fragmentation Management
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
The CUDA caching allocator uses expandable segments, allowing memory blocks to be returned to the OS between training stages rather than staying reserved. This is critical for multi-stage pipelines (Base → LoRA → QLoRA) where each stage needs to reclaim the previous stage's VRAM.

### 6. Between-Stage Cleanup (Python Reference Counting)
A critical gotcha with Python + PyTorch: `del model` inside a function does **not** free CUDA memory if a reference to the object still exists in the calling scope. The correct pattern is:

```python
# WRONG — doesn't free CUDA memory:
def cleanup(model): del model; gc.collect(); torch.cuda.empty_cache()
cleanup(base_model)  # base_model still has refcount > 0 in notebook scope

# CORRECT — zero the notebook-scope reference first:
base_model = None
gc.collect()
torch.cuda.empty_cache()
```

---

## Troubleshooting

### `torch_dtype` deprecated in transformers 5.x
**Error**: `[transformers] 'torch_dtype' is deprecated! Use 'dtype' instead!`  
**Fix**: In `AutoModelForCausalLM.from_pretrained()`, use `dtype=torch.bfloat16` instead of `torch_dtype=torch.bfloat16`.

### `TypeError: unexpected keyword argument 'overwrite_output_dir'`
**Cause**: `overwrite_output_dir` was removed from `TrainingArguments` in transformers 5.x.  
**Fix**: Filter `TrainingArguments` parameters dynamically:
```python
import inspect
valid_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
args = {k: v for k, v in args_dict.items() if k in valid_params}
return TrainingArguments(**args)
```

### `CUDA out of memory: Tried to allocate 6.33 GiB` (between-stage)
**Cause**: transformers 5.x `caching_allocator_warmup` pre-allocates ~6.33 GB when loading with `device_map`. If the previous stage's model is still referenced in notebook scope, this fails.  
**Fix**: Set ALL notebook-scope references to `None` before calling `gc.collect()`:
```python
base_model = None        # critical — function-scope del is not enough
cleanup_torch_objects()  # gc.collect() + torch.cuda.empty_cache()
```

### QLoRA training NaN loss
**Cause**: dtype mismatch between 4-bit base and adapters.  
**Fix**: Always call `prepare_model_for_kbit_training()` before `get_peft_model()`:
```python
base = prepare_model_for_kbit_training(base)  # must come first
model = get_peft_model(base, lora_config)
```

### `bitsandbytes` CUDA not found
**Fix**: Reinstall with correct CUDA:
```bash
uv pip install bitsandbytes --upgrade
```

### Model loads on CPU despite GPU being available
**Cause**: `device_map="auto"` chose CPU because VRAM appeared full.  
**Fix**: Free VRAM before loading, then reload. Check `nvidia-smi` for zombie GPU processes.

---

## Lessons Learned

1. **Python refcounting is the silent VRAM leak.** After any long-running GPU stage, every notebook-scope variable holding the model must be set to `None` before calling `gc.collect()` — a `del` inside a cleanup function is insufficient because the calling scope still holds a reference.

2. **transformers 5.x broke several TrainingArguments fields.** The `inspect.signature` filter pattern (`{k: v for k, v in args_dict.items() if k in valid_params}`) is the correct way to write version-robust training code.

3. **NF4 quantization delivers near-lossless accuracy.** Demo QLoRA (20K samples, 4-bit NF4 base) achieved **94.20% accuracy** vs. **94.20% for demo LoRA** — identical, while using 3.3 GB less VRAM (4.26 GB vs. 7.56 GB peak). The full LoRA run on 110K samples reached 95.15%, suggesting the remaining ~1% gap is a data-size effect, not quantization loss.

4. **The zero-shot baseline is surprisingly strong.** Granite 4.1 3B achieves 89.15% on AG News zero-shot. Fine-tuning is most valuable when the label space is specialized or the format requires exact compliance.

5. **Gradient checkpointing + batch=1 is slow but enables large models.** Effective batch=16 via gradient accumulation preserves optimizer behavior while keeping activation memory minimal.

6. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is non-negotiable** for multi-stage notebooks. Without it, the allocator can't return fragmented blocks between stages, causing OOM even when total VRAM appears sufficient.

---

## Future Improvements

1. **Multi-epoch training**: 2–3 epochs typically yield another 1–2% accuracy gain with diminishing returns after that
2. **Higher rank**: r=32 or r=64 for tasks that require more expressive adapters
3. **Merge adapters**: `model.merge_and_unload()` creates a single merged model for simpler deployment
4. **GPTQ/AWQ quantization of merged model**: Apply post-training quantization for production inference
5. **Other PEFT methods**: IA³, DoRA, LoftQ — compare with LoRA on the same task and dataset
6. **Other datasets**: Same pipeline works for any text classification task; try TREC, SST-5, or a custom domain dataset
7. **Flash Attention 2**: Would reduce attention memory from O(n²) to O(n), enabling longer sequences or larger batches
8. **Evaluate on the full test set**: Current evaluation uses 2000 samples for speed; full 7,600-sample test set would give more reliable estimates

---

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- [PEFT: State-of-the-Art Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft) — HuggingFace
- [IBM Granite 4.1 3B Base](https://huggingface.co/ibm-granite/granite-4.1-3b-base) — IBM Research
- [AG News Dataset](https://huggingface.co/datasets/fancyzhx/ag_news) — Zhang et al., 2015
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) — Dettmers et al.
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [HuggingFace Accelerate](https://github.com/huggingface/accelerate)

---

*Developed and tested on Ubuntu Linux with NVIDIA RTX 4060 Laptop GPU (8 GB VRAM), CUDA 13.1.*
