import json

# Characteristic Harry Potter words
hp_unique_words = [
    "dumbledore", "voldemort", "snape", "hagrid", "hermione", "ron", "malfoy", "sirius", "lupin", "mcgonagall",
    "neville", "dobby", "bellatrix", "umbridge", "quirrell", "hogwarts", "azkaban", "diagon", "hogsmeade",
    "grimmauld", "durmstrang", "beauxbatons", "ministry", "forbidden", "ollivanders", "expelliarmus", "avada",
    "crucio", "lumos", "alohomora", "expecto", "imperio", "stupefy", "accio", "horcrux", "patronus", "horcuxes",
    "thestral", "hippogriff", "nagini", "basilisk", "portkey", "time-turner", "pensieve", "gryffindor", "slytherin",
    "ravenclaw", "hufflepuff", "quidditch", "muggle"
]

# Normalize to lowercase (if needed)
hp_unique_words = [word.lower() for word in hp_unique_words]

# Output file path
output_file = "words_to_forget.json"

# Write to JSON file
with open(output_file, "w") as f:
    json.dump(hp_unique_words, f, indent=2)

print(f"Saved {len(hp_unique_words)} words to {output_file}")
