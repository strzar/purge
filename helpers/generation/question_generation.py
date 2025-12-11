import os
import argparse
import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

model_hf_path = "meta-llama/Llama-3.2-3B-Instruct"
model_name = "Llama-3.2-3B-Instruct"
reject = f"Reject_{model_name}.json"
output_dir = f"/home/stratis/unlearning/data/rwku_style/{model_name}"

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto')
    device = "cuda"

    # Step 1: Load all_intro.json, generate questions, and write out Questions_<model_name>.json
    with open('all_intro.json', 'r') as file:
        dataset = json.load(file)
        all_entries = []
        for subject, intros in dataset.items():
            questions = []
            # We pick a random intro for each of 300 questions
            for _ in range(300):
                intro = random.choice(intros)
                prompt_messages = [
                    {
                        "role": "user",
                        "content": "{}\nPlease generate a question about {} based on what you know about {}\n"
                            .format(intro, subject, subject)
                    },
                ]
                prompt = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompt += "Question:".format(subject)
                prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)
                terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
                response = model.generate(
                    **prompt_inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                    eos_token_id=terminators,
                )
                start_idx = prompt_inputs['input_ids'].shape[-1]
                question_text = tokenizer.decode(
                    response[0][start_idx:], 
                    skip_special_tokens=True
                ).split('\n')[0].strip()
                questions.append(question_text)

            all_entries.append({
                'subject': subject,
                'intro': intro,
                'questions': questions
            })

    # Write the entire list of entries as one JSON array
    with open(f'Questions_{model_name}.json', 'w') as outfile:
        json.dump(all_entries, outfile, indent=2)

    # Step 2: Load idontknow.jsonl into a list
    idontknow = []
    with open('idontknow.jsonl', 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break
            idontknow.append(line.strip())

    # Step 3: Read back Questions_<model_name>.json as a single JSON array,
    # then iterate over its elements (Option A)
    with open(f'Questions_{model_name}.json', 'r') as file:
        all_entries_loaded = json.load(file)

    cnt = 0
    for data in all_entries_loaded:
        cnt += 1
        subject = data['subject']
        questions = data['questions']
        intro = data['intro']

        output = []
        for i, question_text in enumerate(questions):
            output.append({
                'input': '',
                'output': idontknow[i % len(idontknow)],
                'subject': subject,
                'instruction': question_text,
                'intro': intro,
            })

        # Create a directory named "<count>_<subject>" under the output directory
        subject_dir = os.path.join(
            args.output_dir,
            f"{cnt}_{subject.replace(' ', '_')}"
        )
        os.makedirs(subject_dir, exist_ok=True)

        # Write the "reject" file inside that directory
        with open(os.path.join(subject_dir, args.reject_file), 'w') as f:
            json.dump(output, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=model_hf_path,
    )
    parser.add_argument(
        "--reject_file",
        type=str,
        default=reject,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=output_dir,
    )
    args = parser.parse_args()
    main(args)
