# PURGE: A GRPO-Based Unlearning Method

This repository contains a Python script to fine-tune a causal language model using Group Relative Policy Optimization (GRPO) to “unlearn” specific words or phrases. It leverages Hugging Face’s `transformers` and `datasets` libraries together with the TRL (Transformer Reinforcement Learning) package.

---

## Table of Contents

* [Features](#features)
* [Requirements](#requirements)
* [Installation](#installation)
* [Project Structure](#project-structure)
* [Configuration](#configuration)
* [Usage](#usage)
* [How It Works](#how-it-works)
* [Customizing the Forget List and Dataset](#customizing-the-forget-list-and-dataset)
* [Training Arguments](#training-arguments)
* [Contributing](#contributing)
* [License](#license)

---

## Features

* Loads a list of “Forbidden Token Sequences” (phrases to be unlearned) from JSON.
* Defines a reward function that penalizes any generated completion containing those words.
* Uses `trl.GRPOTrainer` to fine-tune a Hugging Face causal LM (e.g. `microsoft/Phi-3-mini-4k-instruct`).
* Supports GPU (CUDA) or CPU fallback.
* Configurable via constants at the top of the script.

## Requirements

* Python 3.8+
* [`torch`](https://pytorch.org/)
* [`datasets`](https://github.com/huggingface/datasets)
* [`transformers`](https://github.com/huggingface/transformers)
* [`trl`](https://github.com/huggingface/trl)

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/strzar/purge.git
   cd unlearning
   ```

2. **Create and activate a virtual environment**

   ```bash
   conda create -n purge
   conda activate purge
   pip install -r requirements.txt
   ```

3. **Install dependencies**

   ```bash
   pip install torch datasets transformers trl
   ```

## Project Structure

```plaintext
├── README.md
├── grpo.py      # Main training script
├── PURGE/
│   └── 1_Stephen_King/
│       ├── fts.json         # List of forbidden token sequences (JSON array)
│       └── qa_pairs.json    # Forget Dataset of (prompt, completion) pairs
└── outputs/
    └── Phi-3-mini-4k-instruct-1_Stephen_King/  # Checkpoints & logs
```

## Configuration

At the top of `grpo.py`, adjust the following constants:

| Constant              | Description                                                               |
| --------------------- | ------------------------------------------------------------------------- |
| `TARGET`              | Name of the forget target (e.g. `"1_Stephen_King"`)                       |
| `HF_MODEL_NAME`       | Hugging Face model identifier (e.g. `"microsoft/Phi-3-mini-4k-instruct"`) |
| `OUTPUT_DIR`          | Local path for saving fine-tuned model checkpoints                        |
| `FORGET_WORDS_FILE`   | Path to JSON file containing an array of words/phrases to forget          |
| `FORGET_DATASET_FILE` | Path to JSON file containing QA pairs used for training                   |

## Usage

Run the training script directly:

```bash
python grpo.py
```

* The script will detect GPU if available, otherwise it will use CPU.
* It loads the first 100 examples from your QA dataset (adjustable via `dataset.select(range(100))`).
* Training logs, checkpoints, and the final model will be saved under `OUTPUT_DIR`.

## How It Works

1. **Load Forget Words**
   Reads `fts.json` into a Python `set` for fast matching.

2. **Define Reward Function**

   ```python
   ```

def reward\_func(completions, \*\*kwargs):
pattern = re.compile(
r'\b(?:' + '|'.join(map(re.escape, forget\_words)) + r')\b',
re.IGNORECASE
)
return \[0.0 if pattern.search(c) else 1.0 for c in completions]

```
   - If any forbidden word appears in a generated completion, that sample’s reward is 0.0; otherwise 1.0.

3. **Prepare Dataset**  
   Loads your QA JSON file into a Hugging Face `Dataset`, shuffles/splits if desired, and selects a subset.

4. **Configure and Launch GRPO**  
   Uses `trl.GRPOConfig` to specify training hyperparameters, then instantiates `GRPOTrainer` and calls `.train()`.

Customizing the Forget List and Dataset
---------------------------------------

- **Forget List (`fts.json`)**: Add or remove words/phrases you wish the model to unlearn.
- **QA Dataset (`qa_pairs.json`)**: Curate the prompts and target completions used to guide the unlearning process.  

You can also modify the `dataset.select(range(100))` call to include more or fewer examples, or implement a proper train/test split using `dataset.train_test_split()`.

Training Arguments
------------------

Adjust these in the `GRPOConfig` block:

| Argument                      | Default | Description                                       |
|-------------------------------|---------|---------------------------------------------------|
| `num_train_epochs`            | 40      | Number of epochs to train                         |
| `per_device_train_batch_size` | 2       | Batch size per GPU/CPU                            |
| `gradient_accumulation_steps` | 8       | Steps to accumulate gradients before update       |
| `num_generations`             | 4       | Number of generations per input for reward eval   |
| `logging_steps`               | 10      | Log training metrics every N steps                |
| `save_strategy`               | steps   | When to save checkpoints (`"steps"` or `"epoch"`) |
| `save_steps`                  | 500     | Save checkpoint every N steps                     |
| `save_total_limit`            | 1       | Max number of saved checkpoints                   |

Contributing
------------

Contributions, issues, and feature requests are welcome!  
1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/YourFeature`)  
3. Commit your changes (`git commit -m 'Add YourFeature'`)  
4. Push to the branch (`git push origin feature/YourFeature`)  
5. Open a Pull Request

License
-------

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

```
