from huggingface_hub import create_repo, upload_folder
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Unlearning target name, e.g. '1_Stephen_King'",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    TARGET = args.target
    repo_id = f"strzara/Phi-3-mini-4k-instruct-{TARGET}_PURGE"

    # Create repo as PUBLIC
    create_repo(repo_id, repo_type="model", private=False, exist_ok=True)

    # Upload the finetuned unlearning model folder
    upload_folder(
        repo_id=repo_id,
        folder_path=f"/path/to/models/binary/Phi-3-mini-4k-instruct-{TARGET}",#here is missing a /checkpoint-1000
        commit_message="Upload finetuned unlearning model"
    )

if __name__ == '__main__':
    main()
