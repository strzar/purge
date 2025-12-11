import os
from datasets import load_dataset

def main():
    dataset_name = "locuslab/TOFU"

    # All dataset configs from the image
    configs = [
        "full",
        "forget01",
        "forget01_perturbed",
        "forget05",
        "forget05_perturbed",
        "forget10",
        "forget10_perturbed",
        "holdout01",
        "holdout05",
        "holdout10",
        "real_authors",
        "real_authors_perturbed",
        "retain90",
        "retain95",
        "retain99",
        "retain_perturbed",
        "world_facts",
        "world_facts_perturbed"
    ]

    save_dir = "./TOFU"
    os.makedirs(save_dir, exist_ok=True)

    for config in configs:
        print(f"\n=== Loading config: {config} ===")
        dataset = load_dataset(dataset_name, config)

        # Each config appears as a single split named "train"
        split = "train"
        output_file = os.path.join(save_dir, f"{config}.json")

        print(f"Saving to: {output_file}")
        dataset[split].to_json(
            output_file,
            orient="records",
            lines=False   # standard JSON
        )

    print("\nAll configs downloaded and saved successfully!")

if __name__ == "__main__":
    main()
