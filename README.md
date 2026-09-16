# GPT-2 Reproduction — Distributed Training and Custom LayerNorm

A PyTorch reproduction of GPT-2 with approximately 124 million parameters.

I developed this project to understand the full path from transformer
code to distributed training on rented NVIDIA GPUs, including H100s.
The work covers FineWeb-Edu data preparation, DDP configuration, GPU
optimisation, experiment tracking, and a custom LayerNorm implementation.

The project follows Andrej Karpathy’s
[GPT-2 reproduction](https://github.com/karpathy/build-nanogpt).
I extended the training workflow and documented the implementation in
detailed notebooks. I also studied the Python reference code from
[llm.c](https://github.com/karpathy/llm.c).

This work became the foundation for
[GPT-Valkyrie](https://github.com/Ice-Citron/GPT-Valkyrie),
my subsequent research on normalisation in decoder-only transformers.

**Development period:** July–August 2024  

- [Custom LayerNorm training code](Training%20Code%20-%20Layer%20Norm%20Impl/train_gpt2.py)
- [Training code with Weights & Biases](Training%20Code%20-%20WandB/train_gpt2.py)
- [Implementation notes](GPT2_implementation_Notes.ipynb)
- [Distributed training](#distributed-training)
- [Run the project](#run-the-project)
- [Later research results](#later-research-results)

## What I worked on

### Model implementation and analysis

I worked through the GPT-2 model and its training loop in detail.
The notebooks record the code, tensor shapes, equations, and questions
that arose during the reproduction.

The work includes:

- Token embeddings and learned position embeddings.
- Causal multi-head self-attention.
- Feedforward networks and residual connections.
- Layer normalisation before each transformer sublayer.
- Weight sharing between token embeddings and the output head.
- Parameter initialisation and residual scaling.
- Autoregressive loss and next-token targets.
- GPT-2 checkpoint structure and parameter shapes.

### Cloud GPU training

I used rented NVIDIA GPU instances for the training work.
This included H100 hardware and distributed runs across multiple GPUs.

The saved notebooks document practical environment setup:

- Python runtime and development-header configuration.
- PyTorch and CUDA package installation.
- GPU checks with `nvidia-smi`.
- Dataset preparation and local token shards.
- Four-process launches with `torchrun`.
- Training logs and experiment tracking.

### Distributed training configuration

I worked with PyTorch DistributedDataParallel, or DDP.
The configuration connects GPU count, batch size, sequence length, and
gradient accumulation.

I examined how each process receives its data.
I also studied when processes must synchronise their gradients.

### Experiment tracking

I added Weights & Biases integration to the training workflow.
The custom LayerNorm version records performance measurements alongside
the loss values.

These records help compare model changes and inspect training behaviour.

### Custom LayerNorm

I replaced `nn.LayerNorm` with an explicit implementation of the
normalisation equation.

The replacement applies to both normalisation layers in every
transformer block. It also applies to the final normalisation layer.

This experiment led to the broader normalisation study in GPT-Valkyrie.

## Model architecture

The main training scripts construct the model from random initialisation.

| Property | Configuration |
|---|---|
| Architecture | GPT-2, decoder-only transformer |
| Model size | Approximately 124 million parameters |
| Transformer blocks | 12 |
| Attention heads per block | 12 |
| Hidden dimension | 768 |
| Head dimension | 64 |
| Feedforward dimension | 3,072 |
| Context length | 1,024 tokens |
| Tokenizer | GPT-2 byte-pair encoding through `tiktoken` |
| Tokenizer vocabulary | 50,257 tokens |
| Training vocabulary dimension | 50,304 |
| Position representation | Learned position embeddings |
| Activation | GELU with the `tanh` approximation |
| Normalisation placement | Before attention and before the feedforward network |
| Output normalisation | Final LayerNorm |
| Output head | Linear projection with shared token-embedding weights |
| Objective | Next-token cross-entropy loss |

The training vocabulary dimension is padded to 50,304.
The tokenizer retains its original 50,257 token IDs.

Each transformer block follows this structure:

```text
Input
  │
  ├── LayerNorm → Causal self-attention → Residual addition
  │
  └── LayerNorm → Feedforward network  → Residual addition
  │
Output
```

The model applies a final LayerNorm after the twelve blocks.
The output head then produces logits for the next token.

## Data preparation

The training corpus is
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
with the `sample-10BT` configuration.

The `fineweb.py` script prepares the data:

1. Download the selected dataset.
2. Add an end-of-text token before each document.
3. Encode the document with the GPT-2 tokenizer.
4. Store token IDs as NumPy `uint16` arrays.
5. Write token shards to `edu_fineweb10B`.

The script uses multiprocessing for tokenisation.
Each full shard contains 100 million tokens.

The first shard serves as the validation split.
The remaining shards serve as the training split.

### Batch construction

`DataLoaderLite` reads the token shards and constructs input-target pairs.

For a batch with `B` sequences and a sequence length of `T`:

- The loader reads `B × T + 1` tokens.
- The input contains the first `B × T` tokens.
- The target contains the same sequence, shifted by one token.

This shift gives each input position its next-token target.

The loader converts the stored `uint16` values through NumPy `int32`
before it creates PyTorch integer tensors.

The data shards are generated files. They are not included in this
repository.

## Distributed training

The distributed path uses PyTorch DDP with the NCCL backend.

`torchrun` starts one process for each GPU.
The script reads `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` to configure
each process.

| Component | Behaviour |
|---|---|
| GPU assignment | Each process selects its local CUDA device |
| Model replication | Each process holds a model replica |
| Data assignment | Each rank starts at a different token position |
| Loader advancement | Each rank advances by the combined batch width |
| Gradient accumulation | Several micro-batches contribute to one optimiser update |
| Gradient synchronisation | DDP synchronises on the final accumulation micro-step |
| Loss reporting | Distributed reductions combine loss measurements |
| Experiment output | Rank zero writes metrics and model checkpoints |

### Global batch size

The scripts target **524,288 tokens per optimiser update**.

```text
Global token batch
    = sequences per GPU
    × tokens per sequence
    × GPU count
    × accumulation steps
```

With the saved configuration:

```text
524,288 = 64 × 1,024 × 4 × 2
```

The code calculates the accumulation count from the process count.

| GPUs | Sequences per GPU | Tokens per sequence | Accumulation steps |
|---:|---:|---:|---:|
| 1 | 64 | 1,024 | 8 |
| 2 | 64 | 1,024 | 4 |
| 4 | 64 | 1,024 | 2 |
| 8 | 64 | 1,024 | 1 |

This keeps the global token batch constant across these configurations.

The code checks that the division produces an integer.
A change to GPU count or micro-batch size must preserve that condition.

### Accumulation and synchronisation

Each micro-batch loss is divided by the accumulation count before
backpropagation. This preserves the mean-loss scale across the update.

The script controls `require_backward_grad_sync` before the forward pass.
Only the final micro-step requires DDP gradient synchronisation.

After accumulation, the script clips the global gradient norm.
It then applies one optimiser update.

## GPU optimisation

The training code combines several optimisation techniques.

| Technique | Purpose |
|---|---|
| `torch.compile` | Compile the model execution path |
| `bfloat16` autocast | Use reduced precision for supported operations |
| High float32 matrix-multiplication precision setting | Permit faster internal matrix arithmetic where supported |
| PyTorch scaled dot-product attention | Use efficient attention kernels with causal masking |
| Fused AdamW on CUDA | Use the fused optimiser when available |
| Vocabulary padding | Align the training vocabulary dimension |
| Gradient accumulation | Reach the target token batch with smaller micro-batches |
| Deferred DDP synchronisation | Reduce communication during accumulation |

The attention implementation uses
`torch.nn.functional.scaled_dot_product_attention`.
PyTorch selects the attention backend for the device and input.

The training loop reports step duration and tokens per second.
The custom LayerNorm version also sends these measurements to
Weights & Biases.

## Training configuration

The following values come from the custom LayerNorm training script.

| Setting | Value |
|---|---|
| Micro-batch size per GPU | 64 sequences |
| Sequence length | 1,024 tokens |
| Global batch size | 524,288 tokens |
| Maximum optimiser steps | 19,073 |
| Planned token budget | Approximately 10 billion tokens |
| Optimiser | AdamW |
| AdamW beta values | `0.9`, `0.95` |
| AdamW epsilon | `1e-8` |
| Weight decay | `0.1` |
| Maximum learning rate | `6e-4` |
| Minimum learning rate | `6e-5` |
| Warmup | 715 steps |
| Learning-rate schedule | Linear warmup, then cosine decay |
| Gradient norm limit | `1.0` |
| Training random seed | `1337` |
| Validation interval | Every 250 steps and at the final evaluation |
| Validation batches per process | 20 |
| Checkpoint interval | Every 5,000 steps and at the final evaluation |
| Compilation | Enabled in the W&B and custom LayerNorm versions |

The optimiser applies weight decay to matrices and embedding tables.
It excludes one-dimensional parameters, such as biases and normalisation
parameters.

The model also scales the initial weights of residual output projections.
This accounts for the number of transformer blocks.

## Custom LayerNorm

The custom implementation is in
[Training Code - Layer Norm Impl](Training%20Code%20-%20Layer%20Norm%20Impl/).

For each token, the implementation computes the mean and variance across
the hidden dimension.

$$
\operatorname{LayerNorm}(x)
=
\gamma \odot
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
+
\beta
$$

| Component | Implementation |
|---|---|
| Mean | `x.mean(dim=-1, keepdim=True)` |
| Variance | `torch.var(..., unbiased=False)` |
| Numerical constant | `eps = 1e-5` |
| Learned scale | `gain`, initialised to one |
| Learned offset | `bias`, initialised to zero |
| Gradients | PyTorch autograd |

The implementation uses population variance.
It normalises the final dimension of the input tensor.

I applied the replacement at three locations:

- `ln_1`, before attention in each block.
- `ln_2`, before the feedforward network in each block.
- `ln_f`, after the final transformer block.

The research notebooks examine the mathematical role of normalisation.
They also compare LayerNorm with BatchNorm and other proposed methods.

This work established the code and mathematical basis for the later
LayerNorm, RMSNorm, and PowerNorm experiments.

## Evaluation and experiment records

### Validation loss

The training loop evaluates a separate validation shard.
It averages loss across twenty batches per process.

In a distributed run, the processes combine their validation measurements.
Rank zero records the result.

### HellaSwag

The repository includes HellaSwag evaluation code.

For each example, the evaluator:

1. Combines the context with each of four candidate completions.
2. Calculates token losses for each candidate.
3. Selects the completion tokens with a mask.
4. Averages the loss over those tokens.
5. Chooses the candidate with the lowest average loss.

Each DDP rank evaluates a different subset of examples.
Distributed reductions combine the correct-answer and example counts.

### Text generation

The training script also contains periodic text generation.

It uses:

- The prompt `Hello, I'm a language model,`.
- Top-k sampling with `k = 50`.
- Four generated sequences per process.
- A maximum total sequence length of 32 tokens.

The sample generator uses a rank-specific random seed.

### Compilation and evaluation

The W&B and custom LayerNorm scripts set `use_compile = True`.

With this setting, validation loss remains active.
The scripts skip HellaSwag evaluation and text generation.

Set `use_compile = False` in the selected script to enable those paths.

### Weights & Biases

The custom LayerNorm version records:

- Training loss.
- Validation loss.
- Learning rate.
- Training step.
- Global gradient norm.
- Step duration.
- Tokens per second.
- HellaSwag measurements when that evaluation path is active.

Rank zero creates the W&B run.
The script prints its run name and run ID.

### Local outputs and checkpoints

The scripts write local records under `log/`.

| Output | Contents |
|---|---|
| `log/log.txt` | Training loss, validation loss, and enabled benchmark results |
| `log/model_*.pt` | Model state and checkpoint metadata |
| W&B run | Configuration and logged measurements |

Each model checkpoint contains:

- The model state dictionary.
- The model configuration.
- The current step.
- The validation loss.

These checkpoints do not contain the optimiser state or data-loader
position. The scripts do not implement a complete automatic training
resume.

Each launch clears `log/log.txt`.
Preserve existing run outputs before a new launch.

## Repository guide

| Location | Contents |
|---|---|
| [Training Code - Layer Norm Impl](Training%20Code%20-%20Layer%20Norm%20Impl/) | Custom LayerNorm, distributed training, and extended W&B metrics |
| [Training Code - WandB](Training%20Code%20-%20WandB/) | Training version with PyTorch LayerNorm and W&B integration |
| [Training Code - Original](Training%20Code%20-%20Original/) | Earlier working versions and an adapted llm.c training script |
| [Original - nanoGPT](Original%20-%20nanoGPT/) | Reference files for Karpathy’s GPT-2 reproduction |
| [Original - LLM_C](Original%20-%20LLM_C/) | Python training reference from llm.c |
| [Original - LLM_C Layer Norm](Original%20-%20LLM_C%20Layer%20Norm/) | LayerNorm reference code |
| [GPT2 implementation notes](GPT2_implementation_Notes.ipynb) | Detailed model and training-loop study |
| [Additional notes, V1](GPT2%20Implementation%20%5BAdditional-Notes%5D%20V1.ipynb) | Gradient norms, regularisation, weight decay, and accumulation |
| [Additional notes, V2](GPT2%20Implementation%20%5BAdditional_Notes%5D%20V2.ipynb) | Distributed loaders, shard formats, and implementation questions |
| [Normalisation research, V1](LN%20-%20EE%20F%20%5BResearch_Notes%5D%20V1.ipynb) | LayerNorm and BatchNorm equations and examples |
| [Normalisation research, V2](LN%20-%20EE%20F%20%5BResearch_Notes%5D%20V2.ipynb) | Further normalisation comparisons and research questions |
| [Normalisation research, V3](LN%20-%20EE%20F%20%5BResearch_Notes%5D%20V3.ipynb) | Further notes on normalisation behaviour |
| [Notes.md](Notes.md) | Dated links to study conversations |

The repository preserves the development sequence.
The earlier working copies contain intermediate changes.

Start with `Training Code - Layer Norm Impl` for the custom normalisation
experiment. Use `Training Code - WandB` for the version with
`nn.LayerNorm`.

### Notebook experiments

The `play.ipynb` notebooks contain smaller checks that support the
training work.

These include:

- Inspection of GPT-2 checkpoint tensors.
- Position-embedding plots.
- Manual text generation.
- Input-target batch construction.
- Shared-weight checks.
- Residual-scale experiments.
- Gradient-accumulation examples.
- Training-log plots.

The implementation notes also examine the differences between NumPy
shards and the binary formats used by llm.c.

## Run the project

The instructions below use a Linux host with NVIDIA GPUs.
The distributed training path requires CUDA and NCCL.

The saved notebooks record Python 3.10.14 and the following core packages.

| Package | Recorded version |
|---|---|
| PyTorch | `2.3.0` |
| Transformers | `4.41.2` |
| Datasets | `2.20.0` |
| NumPy | `1.26.4` |
| PyArrow | `16.0.0` |
| Requests | `2.32.3` |
| Weights & Biases | `0.17.5` |
| Matplotlib | `3.9.1` |

These are historical environment records.
The repository does not contain a complete dependency lockfile.

### 1. Get the repository

```bash
git clone https://github.com/Ice-Citron/GPT2-Reproduction.git
cd GPT2-Reproduction
```

### 2. Prepare the environment

Use Python 3.10 for the recorded environment.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the CUDA 12.1 build of PyTorch 2.3.0:

```bash
python -m pip install torch==2.3.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

Install the training dependencies:

```bash
python -m pip install \
  "numpy==1.26.4" \
  "transformers==4.41.2" \
  "datasets==2.20.0" \
  "pyarrow==16.0.0" \
  "requests==2.32.3" \
  "wandb==0.17.5" \
  tiktoken tqdm
```

For the analysis notebooks, also install:

```bash
python -m pip install "matplotlib==3.9.1" jupyter
```

### 3. Prepare FineWeb-Edu

Enter the custom LayerNorm directory:

```bash
cd "Training Code - Layer Norm Impl"
```

Prepare the token shards:

```bash
python fineweb.py
```

This command prepares the full `sample-10BT` subset.
The token payload alone requires approximately 20 GB.
The dataset cache and other files require additional space.

The script writes the shards to `edu_fineweb10B` beside `fineweb.py`.

### 4. Configure experiment tracking

For an online W&B run:

```bash
wandb login
```

The script uses the project name `nanoGPT-Valkyrie`.
This historical name remains in the code.

For local W&B records, set offline mode before the launch:

```bash
export WANDB_MODE=offline
```

### 5. Start training

For four GPUs:

```bash
torchrun --standalone --nproc_per_node=4 train_gpt2.py
```

For one GPU:

```bash
python train_gpt2.py
```

Run these commands from the selected training directory.
The data loader resolves `edu_fineweb10B` from the current directory.

The four-GPU configuration uses two accumulation steps.
The one-GPU configuration uses eight.

### 6. Change the configuration

The main `train_gpt2.py` scripts define their settings in the source file.

Edit the relevant values for:

- Micro-batch size, `B`.
- Sequence length, `T`.
- Global token batch, `total_batch_size`.
- Learning-rate limits and warmup.
- Maximum training steps.
- Compilation, `use_compile`.

The W&B configuration dictionary also records these settings.
Update the recorded configuration when you change the runtime values.

The separate llm.c-derived script has a different argument interface.

## Later research results

This reproduction led to
[GPT-Valkyrie](https://github.com/Ice-Citron/GPT-Valkyrie),
the subsequent normalisation study.

That study extended the training foundation with:

- LayerNorm, RMSNorm, and PowerNorm comparisons.
- Normalisation ablations in pretrained models.
- BillSum and SQuAD fine-tuning.
- Text-generation evaluation.
- Statistical comparisons across model variants.

The figures below come from that later study.

### Normalisation ablation

![Normalisation variants from the later GPT-Valkyrie study](docs/images/ablation-variants.png)

*Four normalisation layouts from GPT-Valkyrie. The final normalisation
layer remains in every variant.*

### BillSum fine-tuning

![BillSum loss from the later GPT-Valkyrie study](docs/images/billsum-loss.png)

*The eight LayerNorm and RMSNorm variants converge to similar BillSum
training loss values.*

The later study also reported approximately 6% higher pre-training
throughput with RMSNorm than with LayerNorm.

Read the full methods and results:

- [GPT-Valkyrie research repository](https://github.com/Ice-Citron/GPT-Valkyrie)
- [Research paper](https://github.com/Ice-Citron/GPT-Valkyrie/blob/main/Extended%20Essay%20-%20Transformers.pdf)
- [Paper backup on Google Drive](https://drive.google.com/file/d/1dlhTgv4-A2cCYSsL00An_XpfGpg1DyWy/view)
- [Published research checkpoints](https://github.com/Ice-Citron/GPT-Valkyrie#model-checkpoints)

## Related projects

These repositories record the broader development of this work.

| Project | Focus |
|---|---|
| [GPTesla-Code-Generation](https://github.com/Ice-Citron/GPTesla-Code-Generation) | Python code generation, a custom tokenizer, and distributed training with Accelerate |
| [GPT-Foundations](https://github.com/Ice-Citron/GPT-Foundations) | GPT architecture and tokenizer exercises |
| **GPT2-Reproduction** | GPT-2 reproduction, distributed training, and the first custom LayerNorm experiment |
| [GPT-Valkyrie](https://github.com/Ice-Citron/GPT-Valkyrie) | Normalisation research, task fine-tuning, evaluation, and the final paper |

## Acknowledgements

Andrej Karpathy’s teaching and reference code provided the foundation
for this project.

The main sources are:

- [build-nanogpt](https://github.com/karpathy/build-nanogpt):
  the GPT-2 reproduction and accompanying lecture.
- [nanoGPT](https://github.com/karpathy/nanoGPT):
  the broader GPT training implementation.
- [llm.c](https://github.com/karpathy/llm.c):
  the Python reference model and lower-level implementation examples.
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu):
  the pre-training dataset.
- [HellaSwag](https://github.com/rowanz/hellaswag):
  the completion-selection benchmark.

The notebooks preserve my implementation notes and study questions.
Some notes also preserve AI-assisted explanations from the original
study sessions.

## License

This repository uses the [MIT License](LICENSE).
