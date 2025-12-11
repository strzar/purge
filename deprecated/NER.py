import spacy
import json
from tqdm import tqdm
from collections import Counter

nlp = spacy.load("en_core_web_sm")

def split_into_chunks(text, size=100_000):
    return [text[i:i+size] for i in range(0, len(text), size)]

def extract_named_entities(text, chunk_size=100_000):
    labels = {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "EVENT"}
    counter = Counter()
    for chunk in tqdm(split_into_chunks(text, chunk_size)):
        doc = nlp(chunk)
        ents = [ent.text.strip().lower() for ent in doc.ents if ent.label_ in labels]
        counter.update(ents)
    return counter

def main():
    with open('./data/forget.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    freqs = extract_named_entities(text)
    top100 = [e for e, _ in freqs.most_common(100)]
    forget_words = set(top100)
    with open("./data/forget_set.json", "w") as jf:
        json.dump(list(forget_words), jf)

if __name__ == '__main__':
    main()