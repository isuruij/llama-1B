import pandas as pd
from transformers import AutoTokenizer

# 1. Define the test Sinhala texts: a short greeting, and a longer real paragraph
# (pulled verbatim from new_split_v2/test.jsonl, the project's own history-textbook QA data)
# so the comparison isn't based on one short sentence alone.
short_text = "ආයුබෝවන්, ඔයාට කොහොමද? ලංකාව ඉතා ලස්සන රටකි."
long_text = (
    "1602 වර්ෂයෙන් පසු පෙරදිග ඕලන්ද වෙළෙඳ සමාගම මූල්‍යමය වශයෙන් ද ශක්තිමත් වූ හෙයින් "
    "ප්‍රබල නැව් කණ්ඩායම් ආසියාවට එවීමට ලන්දේසීන්ට හැකි විය. මෙසේ 'VOC' සමාගම යටතේ "
    "පැමිණි ලන්දේසීහු ජාවා දූපතේ බෙතාවිය හෙවත් බතාවියේ තම මූලස්ථානය පිහිටුවාගෙන "
    "අග්නිදිග ආසියාතික දූපත් හා ඉන්දියානු වෙරළබඩ ප්‍රදේශයේ ස්ථාන කිහිපයක ස්වකීය බලය "
    "ගොඩනැගූහ."
)

texts_to_test = [
    ("Short sentence", short_text),
    ("Large paragraph", long_text),
]

# 2. Load Tokenizers (Using your HF Token if needed for Base Llama)
HF_TOKEN = "" 

print("Loading tokenizers...")
extended_tokenizer = AutoTokenizer.from_pretrained("polyglots/Extended-Sinhala-LLaMA", token=HF_TOKEN)
base_llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token=HF_TOKEN)
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", token=HF_TOKEN)
extended_qwen_tokenizer = AutoTokenizer.from_pretrained("isji/Extended-Sinhala-Qwen3", token=HF_TOKEN)

# 3. Process the entire sentence for each model
tokenizers_list = [
    ("Extended Sinhala (Polyglots)", extended_tokenizer),
    ("Base Llama 3", base_llama_tokenizer),
    ("Qwen 4B", qwen_tokenizer),
    ("Extended Sinhala Qwen3", extended_qwen_tokenizer),
]

for text_label, test_text in texts_to_test:
    print(f"\n{'#' * 60}")
    print(f"TEXT: {text_label}")
    print(f"Original Text: {test_text}")
    print(f"Total Characters: {len(test_text)}")
    print("#" * 60)

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
