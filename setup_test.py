import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import csv 

os.environ['HF_HOME'] = 'D:/huggingface_cache'

model_id = "google/gemma-3-1b-it"
print(f"Attempting to load model: {model_id}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

try:
    """ folder_path = "malayalam_dataset"
    file_paths = [os.path.join(folder_path, f"data-{i}.parquet") for i in range(1, 37)]
 """
    dataset = load_dataset("ai4bharat/sangraha", data_dir="verified/mal")
    subset = dataset['train'].select(range(100))
    
    print(f"Successfully Loaded {len(subset)} documents \n")
except Exception as e:
    print(f"Failed to load documents: {e}")
    exit()

with open('generated_queries2.csv', 'w', newline='', encoding='utf-8') as csvfile:
    
    writer = csv.writer(csvfile)
    writer.writerow(['document_text', 'generated_query'])

    print("\n Starting query generation and saving results...")

    for i, item in enumerate(subset):
        document_text = item['text']
        print(f"Processing document {i+1}...")
        print(f"Document: {document_text}")
        example_document = "ദുബായ് : ടി20 ലോകകപ്പിൽ ബംഗ്ലാദേശിന്റെ ശേഷിക്കുന്ന മത്സരങ്ങളിൽ നിന്ന് ഷാക്കിബ് അൽ ഹസൻ പരിക്ക് കാരണം പുറത്തായി."
        example_query = "ടി20 ലോകകപ്പിൽ നിന്ന് ഷാക്കിബ് അൽ ഹസൻ പുറത്താകാൻ കാരണമെന്ത്?"
        prompt = (
            "You are a query generation assistant. Your task is to create a single, natural question in the Malayalam language that a user would ask to find the given document. "
            f"--- EXAMPLE ---\n"
            f"DOCUMENT:\n{example_document}\n\n"
            f"USER QUERY:\n{example_query}\n\n"
            f"--- TASK ---\n"
            f"DOCUMENT:\n{document_text}\n\n"
            f"USER QUERY:"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        response = model.generate(**inputs, max_new_tokens=50, num_beams=4)

        input_length = inputs.input_ids.shape[1]
        new_tokens = response[0, input_length:]
        generated_query = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        print(f"  > Generated Query: {generated_query}")
        writer.writerow([document_text, generated_query])

print("\nProcessing complete. Results saved to generated_queries.csv")