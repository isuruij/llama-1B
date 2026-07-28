import pandas as pd
from transformers import AutoTokenizer

# 1. Define the test Sinhala text
test_text = "ආයුබෝවන්, ඔයාට කොහොමද? ලංකාව ඉතා ලස්සන රටකි."

print(f"Original Text: {test_text}")
print(f"Total Characters: {len(test_text)}")
print("=" * 50)

# 2. Load Tokenizers (Using your HF Token if needed for Base Llama)
HF_TOKEN = "" 

print("Loading tokenizers...")
extended_tokenizer = AutoTokenizer.from_pretrained("polyglots/Extended-Sinhala-LLaMA", token=HF_TOKEN)
base_llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token=HF_TOKEN)
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", token=HF_TOKEN)

# 3. Process the entire sentence for each model
tokenizers_list = [
    ("Extended Sinhala (Polyglots)", extended_tokenizer),
    ("Base Llama 3", base_llama_tokenizer),
    ("Qwen 4B", qwen_tokenizer)
]

for name, tokenizer in tokenizers_list:
    # Get the full token list
    full_tokens = tokenizer.tokenize(test_text)
    token_ids = tokenizer.encode(test_text)
    
    print(f"\n[ MODEL: {name} ]")
    print(f"Total Tokens produced: {len(token_ids)}")
    print(f"Efficiency: {round(len(test_text) / len(token_ids), 2)} chars/token")
    print("-" * 40)
    
    # Print EVERY single token generated for the sentence
    # formatted nicely so it doesn't overflow your terminal
    chunk_size = 8
    for i in range(0, len(full_tokens), chunk_size):
        print(full_tokens[i:i+chunk_size])
    print("=" * 50)
