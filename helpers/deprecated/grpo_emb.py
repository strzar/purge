import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
import re
import json

# ============================================================
# Configuration
# ============================================================

TARGET = "1_Stephen_King"
TARGET_CONCEPT = "Stephen King"  # semantic seed concept

HF_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_DIR = f"/mnt/raid5/stratis/models/main/Phi-3-mini-4k-instruct-{TARGET}"
FORGET_WORDS_FILE = f"/home/stratis/unlearning/PURGE/{TARGET}/fts.json"
FORGET_DATASET_FILE = f"/home/stratis/unlearning/PURGE/{TARGET}/qa_pairs.json"
EXPANDED_FORGET_WORDS_FILE = FORGET_WORDS_FILE.replace(".json", "_expanded.json")

# ============================================================
# Step 1: Contextual neighbor retrieval
# ============================================================

def get_contextual_neighbors(model_name, phrase, k=50, contexts=None):
    """
    Get the top-k tokens in the model's embedding space that are semantically
    close to the phrase, using contextual embeddings.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()

    emb = model.get_input_embeddings().weight.detach()

    if contexts is None:
        contexts = [
            "{} is a famous horror author.",
            "{} is known for supernatural and psychological novels.",
            "Books by {} often involve fear and the macabre.",
            "Stories by {} inspired many horror movies."
        ]

    phrase_vecs = []

    for ctx in contexts:
        sentence = ctx.format(phrase)
        inputs = tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # Average last 4 hidden layers for robust semantic meaning
        if hasattr(outputs, "hidden_states"):
            last_hidden = torch.stack(outputs.hidden_states[-4:], dim=0).mean(0)
        else:
            raise ValueError("Model does not return hidden states.")
        vec = last_hidden.mean(dim=1)  # mean over tokens
        phrase_vecs.append(vec)

    phrase_vec = torch.mean(torch.stack(phrase_vecs), dim=0)  # average across contexts
    sims = F.cosine_similarity(phrase_vec, emb, dim=1)
    top_ids = torch.topk(sims, k).indices.tolist()
    tokens = [tokenizer.decode([i]).strip() for i in top_ids]

    # Clean tokens (remove punctuation, subwords, etc.)
    def clean_token(t):
        t = t.strip()
        return t if t.isalpha() and len(t) > 1 else None

    clean_tokens = [t for t in map(clean_token, tokens) if t]
    return clean_tokens

# ============================================================
# Step 2: Expand forget words
# ============================================================

def expand_forget_words(model_name, base_terms, k=50):
    """
    Expand the forget-word list by adding semantically related words
    derived from contextual embeddings.
    """
    all_neighbors = set()
    for term in base_terms:
        print(f"Finding contextual neighbors for '{term}' ...")
        neighbors = get_contextual_neighbors(model_name, term, k=k)
        print(f"Found {len(neighbors)} related tokens: {neighbors[:10]} ...")
        all_neighbors.update(neighbors)
    return all_neighbors

# ============================================================
# Step 3: Reward function
# ============================================================

def make_reward_func(forget_words):
    pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, forget_words)) + r')\b', re.IGNORECASE)

    def reward_func(completions, **kwargs):
        return [0.0 if pattern.search(c) else 1.0 for c in completions]

    return reward_func

# ============================================================
# Step 4: Main Training Routine
# ============================================================

def main():
    # --------------------------------------------------------
    # 1. Load base forget words
    # --------------------------------------------------------
    with open(FORGET_WORDS_FILE, "r") as f:
        base_forget_words = set(json.load(f))
    print(f"Loaded {len(base_forget_words)} base forget words.")

    # --------------------------------------------------------
    # 2. Expand forget words semantically
    # --------------------------------------------------------
    semantic_neighbors = expand_forget_words(HF_MODEL_NAME, [TARGET_CONCEPT], k=50)
    print(f"Expanded forget words to {len(semantic_neighbors)} entries.")

    # Save expanded forget words
    with open(EXPANDED_FORGET_WORDS_FILE, "w") as f:
        json.dump(sorted(list(semantic_neighbors)), f, indent=2)
    print(f"Saved expanded forget words to: {EXPANDED_FORGET_WORDS_FILE}")

    # --------------------------------------------------------
    # 3. Prepare dataset
    # --------------------------------------------------------
    with open(FORGET_DATASET_FILE, "r") as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)
    dataset = dataset.select(range(100))  # limit for debug runs

    # --------------------------------------------------------
    # 4. Reward + Training Config
    # --------------------------------------------------------
    reward_func = make_reward_func(semantic_neighbors)

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=40,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_generations=4,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
    )

    # --------------------------------------------------------
    # 5. Trainer setup
    # --------------------------------------------------------
    trainer = GRPOTrainer(
        model=HF_MODEL_NAME,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=dataset
    )

    # --------------------------------------------------------
    # 6. Train
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Starting GRPO training...")
    trainer.train()
    print("Training completed.")

# ============================================================

if __name__ == "__main__":
    main()
