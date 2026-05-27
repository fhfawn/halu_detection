# Hallucination Detection in Tool Calling

This repository contains an end-to-end pipeline for building and evaluating hallucination detectors in a tool-calling setting. The project starts from ToolACE conversations, injects controlled hallucinations into assistant answers, evaluates pretrained baselines, and fine-tunes a ModernBERT token classifier to detect hallucinated spans.

The main notebook is designed as a full reproducible workflow: if the local dataset folder is empty or incomplete, it can automatically create the required JSONL datasets before running evaluation or training.

## Project overview

Tool-calling assistants should answer using only information supported by the available tool output. In practice, assistants may introduce unsupported facts, contradict tool results, or propose actions that require tools that are not available. This project converts those failures into a span-detection benchmark.

The benchmark covers three hallucination types:

| Type | Description | Example failure pattern |
|---|---|---|
| `hallucination` | A factual span is replaced with a contradictory value | Wrong number, name, date, status, or entity |
| `overgeneration` | An unsupported but plausible sentence is appended | Extra details not present in the tool output |
| `missing_tool` | The answer suggests an action requiring an unavailable tool | Booking, ordering, emailing, scheduling, etc. |

Each example follows a RAGTruth-style schema:

```json
{
  "id": "example_id",
  "query": "user question",
  "context": "tool output or available tool description",
  "output": "assistant answer with injected hallucination",
  "hallucination_labels": [
    {
      "start": 123,
      "end": 145,
      "text": "hallucinated span",
      "type": "overgeneration"
    }
  ]
}
```

## Repository structure

```text
.
├── halu-detection-iman-irina.ipynb        # Full project notebook
├── toolace_full_pipeline_master.ipynb     # Master pipeline notebook, if included
├── toolace_hallucination_inject-2.ipynb   # Dataset-generation notebook, if included
├── toolace_halu_eval.ipynb                # Baseline evaluation notebook, if included
├── toolace_halu_improve.ipynb             # Fine-tuning notebook, if included
├── halu_datasets/                         # Auto-created dataset folder
│   ├── toolace_halu_contradiction.jsonl
│   ├── toolace_halu_overgeneration.jsonl
│   └── toolace_halu_missing_tool.jsonl
├── toolace_halu_eval_results.csv          # Baseline metrics
├── toolace_halu_improve_results.csv       # Final comparison metrics
├── toolace_halu_modernbert_final/         # Optional saved fine-tuned model
├── hallucination_detection_tool_calling_report.md
└── README.md
```

Depending on the submitted version, some intermediate notebooks or generated outputs may not be present until the main notebook is executed.

## Main pipeline

The project consists of four main stages.

### 1. Dataset loading or generation

The notebook defines `ensure_datasets()`, which is the main dataset entry point. It checks whether the expected JSONL files already exist and contain valid rows. If the folder is empty, incomplete, or broken, it calls `generate_datasets()` to create new datasets automatically.

Expected files:

```text
halu_datasets/toolace_halu_contradiction.jsonl
halu_datasets/toolace_halu_overgeneration.jsonl
halu_datasets/toolace_halu_missing_tool.jsonl
```

Dataset generation uses ToolACE as the source dataset and vLLM to create synthetic corruptions. The generated examples contain character-level hallucination spans for evaluation and training.

### 2. Baseline evaluation

Two baseline methods are evaluated:

#### LettuceDetect

A pretrained hallucination detector is used zero-shot. It receives the user query, tool output, and assistant answer, then returns predicted hallucinated spans.

#### LookBackLens

LookBackLens uses attention-based features from a causal language model. For each answer token, it computes how strongly the token attends back to the tool context compared with previous answer tokens. A logistic regression classifier is trained on these token features.

For memory-constrained GPUs, the notebook uses a smaller Qwen backbone for LookBackLens, such as:

```python
LBL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_SEQ = 768
```

### 3. ModernBERT fine-tuning

The improved model is a token-classification version of ModernBERT. Each example is encoded as:

```text
QUESTION: <query>

CONTEXT: <tool output>

ANSWER: <assistant answer>
```

Only answer tokens are supervised. Query, context, and special tokens are masked with `-100`, so they do not contribute to the loss.

Labels:

| Label | Meaning |
|---|---|
| `O` | Non-hallucinated answer token |
| `B-HAL` | Beginning of hallucinated span |
| `I-HAL` | Inside hallucinated span |

The notebook reconstructs negative examples by removing the injected hallucination from positive examples, keeping matched positive/negative pairs in the same split to avoid leakage.

### 4. Final comparison

The final stage evaluates the fine-tuned ModernBERT model using the same character-level and example-level metrics as the baselines, then writes a comparison table to:

```text
toolace_halu_improve_results.csv
```

## Reported results

The final report shows that the fine-tuned ModernBERT model outperforms both baselines, especially on short contradictory hallucinations.

| Dataset | Method | Token F1 | Example F1 | Accuracy |
|---|---:|---:|---:|---:|
| Hallucination | LettuceDetect | 0.191 | 0.678 | 0.638 |
| Hallucination | LookBackLens | 0.308 | 0.656 | 0.492 |
| Hallucination | ModernBERT-FT | **0.599** | **0.841** | **0.861** |
| Overgeneration | LettuceDetect | 0.770 | 0.785 | 0.739 |
| Overgeneration | LookBackLens | 0.814 | 0.745 | 0.635 |
| Overgeneration | ModernBERT-FT | **0.975** | **0.948** | **0.948** |
| Missing tool | LettuceDetect | 0.803 | 0.805 | 0.759 |
| Missing tool | LookBackLens | 0.899 | 0.767 | 0.677 |
| Missing tool | ModernBERT-FT | **0.979** | **0.959** | **0.959** |

## Installation

Create a clean Python environment, then install the main dependencies.

```bash
pip install -U datasets huggingface_hub scikit-learn pandas numpy tqdm
pip install -U torch transformers accelerate safetensors
pip install -U lettucedetect
```

For dataset generation with vLLM:

```bash
pip install -U vllm
```

A pinned setup used during development was:

```bash
pip install -q -U vllm==0.6.3 datasets==2.21.0 huggingface_hub scikit-learn lettucedetect
python -m pip install -q --force-reinstall --no-deps "transformers==4.46.3"
python -m pip install -q --force-reinstall "pyairports==2.1.1"
```

The exact package versions may need adjustment depending on the CUDA runtime and GPU available in the execution environment.

## Hardware notes

This project uses several memory-heavy components. The safest execution strategy is to run generation, baseline evaluation, and fine-tuning as separate stages and clear GPU memory between them.

Recommended memory practices:

```python
import gc
import torch

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
```

For GPUs with about 15 GB VRAM:

- run LettuceDetect on CPU if CUDA compatibility errors occur;
- use `Qwen/Qwen2.5-0.5B-Instruct` for LookBackLens;
- set `MAX_SEQ = 512` or `MAX_SEQ = 768`;
- fine-tune ModernBERT with batch size 1 and gradient accumulation;
- restart the runtime before ModernBERT fine-tuning if Qwen or vLLM was loaded earlier.

A memory-safe ModernBERT setup is:

```python
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 8
gradient_checkpointing = True
fp16 = True
bf16 = False
MAX_LEN = 512
```

## Usage

### Option 1: Run the full notebook

Open the main notebook:

```text
halu-detection-iman-irina.ipynb
```

Then execute the sections in order:

1. environment setup and imports;
2. dataset utilities;
3. dataset generation / `ensure_datasets()`;
4. baseline evaluation;
5. LookBackLens feature extraction;
6. ModernBERT fine-tuning;
7. final comparison and model saving.

### Option 2: Use pre-generated datasets

Place the three JSONL files either in the repository root or in `halu_datasets/`. Then run:

```python
datasets_by_type = ensure_datasets(force_generate=False, min_rows=1)
```

If the files exist and are valid, generation is skipped.

### Option 3: Force dataset regeneration

```python
datasets_by_type = ensure_datasets(force_generate=True, min_rows=1)
```

This reloads ToolACE and regenerates the hallucination datasets.

## Reproducing the benchmark

A typical full run inside the notebook is:

```python
# 1. Load or generate datasets
datasets_by_type = ensure_datasets(force_generate=False, min_rows=1)
data = datasets_by_type

# 2. Build positive/negative pairs
balanced = {}
for name, rows in datasets_by_type.items():
    balanced[name] = []
    for r in rows:
        pos, neg = make_pair(r, name)
        balanced[name].extend([pos, neg])

# 3. Evaluate baselines
# Run LettuceDetect and LookBackLens cells.

# 4. Fine-tune ModernBERT
# Run encoding, training, and evaluation cells.

# 5. Save final comparison
combined.to_csv("toolace_halu_improve_results.csv", index=False)
```

## Outputs

| File or folder | Purpose |
|---|---|
| `halu_datasets/*.jsonl` | Generated hallucination datasets |
| `toolace_halu_eval_results.csv` | Baseline results for LettuceDetect and LookBackLens |
| `toolace_halu_improve_results.csv` | Final comparison including ModernBERT-FT |
| `toolace_halu_modernbert_final/` | Saved fine-tuned model and tokenizer |
| `hallucination_detection_tool_calling_report.md` | Detailed technical report |

## Limitations

The generated benchmark is synthetic, so results may not fully transfer to naturally occurring hallucinations. Contradictory factual hallucinations are short and difficult to localize, while overgeneration and missing-tool examples are often appended sentences and therefore easier to detect. The missing-tool class also mixes factual grounding with tool-availability policy violations, which may require more specialized modeling in future work.

LookBackLens results depend strongly on the causal language model backbone. In low-memory environments, the smaller Qwen model enables reproduction but may underperform larger models.

## Future work

Potential extensions include:

- evaluating on naturally occurring tool-calling hallucinations;
- adding more hallucination categories, such as wrong tool selection or malformed tool arguments;
- using larger attention backbones for LookBackLens on high-memory GPUs;
- adding cross-domain evaluation on tools unseen during training;
- calibrating span thresholds for precision-sensitive or recall-sensitive deployment.

## Authors

Iman Chantieva and Irina Brodskaya

## Repository

<https://github.com/hhfawn/halu_detection>

## dataset and model
<https://huggingface.co/datasets/Fawnnn/Hallucination>
<https://huggingface.co/Fawnnn/ModernBERT-toolace-hallu-detect>
