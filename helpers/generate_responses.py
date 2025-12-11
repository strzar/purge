#!/usr/bin/env python3
"""
Run Phi-3 QA-pair generation for every target listed in NAMES.
Assumes the directory layout:

  /home/stratis/unlearning/RWKU/Target/<NAME>/reject_phi.json
  /home/stratis/unlearning/PURGE/<NAME>/qa_pairs.json
"""

import os
import json
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------
HF_MODEL_NAME   = "microsoft/Phi-3-mini-4k-instruct"
TARGET_BASE_DIR = "/home/stratis/unlearning/RWKU/Target"
OUTPUT_BASE_DIR = "/home/stratis/unlearning/PURGE"

NAMES = [
    # '1_Stephen_King',
    # '2_Confucius',
    # '3_Bruce_Lee',
    # '4_Warren_Buffett',
    # '5_Christina_Aguilera',
    # '6_Cindy_Crawford',
    # '7_Marie_Osmond',
    # '8_Paris_Hilton',
    # '9_Justin_Bieber',
    # '10_Prince_Harry,_Duke_of_Sussex',
    # '11_Miley_Cyrus',
    # '12_Genghis_Khan',
    # '13_Liza_Minnelli',
    # '14_Taylor_Swift',
    # '15_Mark_Cuban',
    # '16_Rhea_Perlman',
    # '17_Mark_Hamill',
    # '18_John_D._Rockefeller',
    # '19_Alanis_Morissette',
    # '20_Marlon_Brando',
    # '21_50_Cent',
    # '22_Jim_Morrison',
    # '23_Evel_Knievel',
    # '24_Beyoncé',
    # '25_Reba_McEntire',
    # '26_Justin_Timberlake',
    # '27_Vanna_White',
    # '28_Lil_Wayne',
    # '29_Anna_Nicole_Smith',
    # '30_Henry_Winkler',
    # '31_Leonardo_da_Vinci',
    # '32_Kanye_West',
    # '33_Paul_Walker',
    # '34_Daniel_Day-Lewis',
    # '35_Jim_Parsons',
    # '36_Henry_Kissinger',
    # '37_Chuck_Norris',
    # '38_Steven_Seagal',
    # '39_Linda_Hamilton',
    # '40_Danny_Trejo',
    # '41_Sam_Elliott',
    # '42_Michael_Strahan',
    # '43_Paul_Simon',
    # '44_Meghan,_Duchess_of_Sussex',
    # '45_Bruce_Springsteen',
    # '46_Raquel_Welch',
    # '47_Lenny_Kravitz',
    # '48_Bob_Saget',
    # '49_Jon_Voight',
    # '50_Ryan_Seacrest',
    # '51_Betty_White',
    # '52_Chris_Brown',
    # '53_Travis_Kelce',
    # '54_Jay-Z',
    # '55_Jackie_Chan',
    # '56_Mark_Harmon',
    # '57_Whitney_Houston',
    # '58_Rihanna',
    # '59_Anderson_Cooper',
    # '60_Brendan_Fraser',
    # '61_Tim_Burton',
    # '62_Serena_Williams',
    # '63_Dionne_Warwick',
    # '64_Michelle_Pfeiffer',
    # '65_Selena_Gomez',
    # '66_Kris_Jenner',
    # '67_Hugh_Laurie',
    # '68_Tom_Clancy',
    # '69_John_Candy',
    # '70_Vin_Diesel',
    # '71_Dakota_Fanning',
    # '72_R._Kelly',
    # '73_Emilio_Estevez',
    # '74_Socrates',
    # '75_Brooke_Shields',
    # '76_Bob_Barker',
    # '77_Val_Kilmer',
    # '78_Jennifer_Lopez',
    # '79_Pamela_Anderson',
    # '80_Tony_Blair',
    # '81_Vincent_van_Gogh',
    '82_Lindsay_Lohan',
    '83_Rebel_Wilson',
    '84_Nicolas_Cage',
    '85_Ted_Danson',
    '86_John_Travolta',
    '87_Robert_Downey_Jr.',
    '88_Jason_Bateman',
    '89_Samuel_L._Jackson',
    '90_Karl_Marx',
    '91_Halle_Berry',
    '92_Larry_Bird',
    '93_Johnny_Cash',
    '94_Chevy_Chase',
    '95_Bill_Paxton',
    '96_Ice_Cube',
    '97_Don_Johnson',
    '98_Dwayne_Johnson',
    '99_RuPaul',
    '100_Matthew_Perry'
]

# ---------------------------------------------
# 2. HELPER FUNCTIONS (same as your original)
# ---------------------------------------------
def generate_response(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    return tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )


def test_model_responses(model, tokenizer, test_prompts):
    qa_pairs = []
    for prompt in tqdm(test_prompts, desc="   ↳ generating"):
        # if prompt == "": continue
        response = generate_response(model, tokenizer, prompt)
        qa_pairs.append({"prompt": prompt, "response": response})
    return qa_pairs


def process_single_target(model, tokenizer, name):
    """Load reject set → generate QA pairs → write output JSON."""
    dataset_path = os.path.join(TARGET_BASE_DIR, name, "reject_phi.json")
    output_path  = os.path.join(OUTPUT_BASE_DIR, name, "qa_pairs.json")

    if not os.path.exists(dataset_path):
        print(f"⚠️  {dataset_path} not found – skipping.")
        return

    with open(dataset_path, "r") as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)

    print(f"• {name} – {len(dataset):,} prompts")
    qa_pairs = test_model_responses(model, tokenizer, dataset["instruction"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=4)

    print(f"   ✓ saved {len(qa_pairs):,} QA pairs → {output_path}\n")


# ---------------------------------------------
# 3. MAIN LOOP
# ---------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load model/tokenizer only ONCE
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_NAME, torch_dtype=torch.float16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model.eval()

    for name in NAMES:
        process_single_target(model, tokenizer, name)

    # Optional: free GPU memory at the end
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
