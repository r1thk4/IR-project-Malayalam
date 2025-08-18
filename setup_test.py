import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import argparse
import json
import re

""" os.environ['HF_HOME'] = 'D:/huggingface_cache' """

def load_model_and_tokenizer(model_id):
    print(f"Attempting to load model: {model_id}\n")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print("Model and tokenizer loaded successfully!\n")
        return tokenizer, model
    except Exception as e:
        print(f"An error occurred while loading model or tokenizer: {e}")
        exit()

""" def load_malayalam_dataset(input_dataset, input_data_dir, start_index, end_index, num):
    print(f"Attempting to load dataset '{input_dataset}' with data_dir='{input_data_dir}'")
    try:
        dataset = load_dataset(input_dataset, data_dir=input_data_dir, split='train')
        total_docs = len(dataset)

        if start_index is None and end_index is None and num is None:
            subset = dataset
            print(f"Successfully Loaded ALL {len(subset)} documents.\n")
            return subset
           
        start = start_index if start_index is not None else 0
        end = end_index if end_index is not None else total_docs

        if num is not None:
            limit_end = start + num
            end = min(end, limit_end)  

        if start < 0 or end > total_docs or start >= end:
            print(f"Invalid range specified or calculated. Final Range: {start} to {end}. Dataset has {total_docs} documents.")
            exit()  

        subset = dataset.select(range(start, end))
        print(f"Successfully Loaded {len(subset)} documents from index {start} to {end}.\n")
            
        return subset
    except Exception as e:
        print(f"Failed to load documents: {e}")
        exit() """

def get_domain_from_category(document_category, prompt_config):
    if document_category in prompt_config:
        return document_category
    
    words = re.split(r'[\s,/&|]+', document_category)
    matched_domains = []

    prompt_keys_lower = {k.lower(): k for k in prompt_config.keys()}
    for word in words:
        word_lower = word.lower()
        if word_lower in prompt_keys_lower:
            og_key = prompt_keys_lower[word_lower]
            if og_key not in matched_domains:
                matched_domains.append(og_key)
    
    if matched_domains:
        return matched_domains
    
    return ['Miscellaneous']


def format_prompt(domains, prompt_config, document_text):
    if isinstance(domains, str):
        domains = [domains]

    all_examples = []
    seen_documents = set() # Use a set to track added documents and avoid duplicates
    valid_domains = []

    for domain in domains:
        if domain in prompt_config:
            valid_domains.append(domain)
            # Add examples from the valid domain
            for example in prompt_config[domain]:
                if example['document'] not in seen_documents:
                    all_examples.append(example)
                    seen_documents.add(example['document'])
        else:
            print(f"Warning: Domain '{domain}' not found in prompt_config.json.")

    if not valid_domains:
        print("No valid domains found. Using 'Miscellaneous' as a fallback.")
        valid_domains = ['Miscellaneous']
        if 'Miscellaneous' in prompt_config:
            for example in prompt_config['Miscellaneous']:
                if example['document'] not in seen_documents:
                    all_examples.append(example)
                    seen_documents.add(example['document'])

    example_prompts = ""
    for example in all_examples:
        example_prompts += f"DOCUMENT: {example['document']}\nUSER QUERY: {example['query']}\n\n"

    domain_string = " and ".join(valid_domains)
    prompt = (
        f"Your task is to create a single, natural question in the Malayalam language that a user in the domain(s) {domain_string} would ask to find the given document. The generated query must be in Malayalam.\n\n"
        "---EXAMPLES---\n"
        f"{example_prompts}"
        "---YOUR TASK---\n"
        f"DOCUMENT:\n{document_text}\n\n"
        "USER QUERY:"
    )
    return prompt

def generate_and_save_queries(tokenizer, model, input_file, output_folder, output_file, prompt_config, max_new_tokens, num_beams, temperature, do_sample, start_index, end_index, num_docs):
    """
    Reads a JSONL file, generates a query for each line, and saves to a new JSONL file.
    """
    os.makedirs(output_folder, exist_ok=True)
    output_jsonl_path = os.path.join(output_folder, f'{output_file}.jsonl')

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
        
        print(f"Starting query generation. Reading from '{input_file}' and saving results to '{output_jsonl_path}'...\n")

        # Read all lines to apply slicing logic
        all_lines = infile.readlines()
        total_docs = len(all_lines)

        start = start_index if start_index is not None else 0
        end = end_index if end_index is not None else total_docs

        if num_docs is not None:
            limit_end = start + num_docs
            end = min(end, limit_end)

        if start < 0 or end > total_docs or start >= end:
            print(f"FATAL: Invalid range. Final Range: {start} to {end}. Input file has {total_docs} documents.")
            exit()

        lines_to_process = all_lines[start:end]
        
        for i, line in enumerate(lines_to_process):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON line at index {start + i}: {line.strip()}")
                continue

            document_text = item.get('text')
            document_category = item.get('category', 'Miscellaneous')
            doc_id = item.get('doc_id')

            print(f"\nProcessing document {start + i + 1}/{total_docs} (Item {i+1}/{len(lines_to_process)})...")
            print(f"Original Category: '{document_category}'")

            if not document_text or not document_text.strip():
                print("Skipping empty document.")
                continue
            
            domains = get_domain_from_category(document_category, prompt_config)
            print(f"Mapped Domain(s): {domains}")

            prompt = format_prompt(domains, prompt_config, document_text)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "num_beams": num_beams,
            }
            if do_sample:
                generation_kwargs["do_sample"] = True
                generation_kwargs["temperature"] = temperature

            response = model.generate(**inputs, **generation_kwargs)

            input_length = inputs.input_ids.shape[1]
            new_tokens = response[0, input_length:]
            generated_query = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            print(f"Generated Query: {generated_query}")

            record = {
                "doc_id": doc_id,
                "text": document_text,
                "category": document_category,
                "generated_query": generated_query
            }
            outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
            outfile.flush()

    print(f"\nProcessing complete. Results saved to {output_jsonl_path}")


def main_func():
    parser = argparse.ArgumentParser(description="Generate synthetic queries for Malayalam documents.")
    
    # --- File Arguments ---
    parser.add_argument("-if", "--input_file", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("-o", "--output_folder", type=str, default="results", help="Folder to save the output.")
    parser.add_argument("-f", "--output_file", type=str, default="generated_queries", help="Base name for the output JSONL file.")
    
    # --- Model and Data Loading Arguments ---
    parser.add_argument("-m", "--model_name", type=str, default="google/gemma-2b-it", help="Hugging Face model ID.")
    parser.add_argument("-s", "--start", type=int, default=None, help="The starting line number of the documents to process.")
    parser.add_argument("-e", "--end", type=int, default=None, help="The ending line number (exclusive) of the documents to process.")
    parser.add_argument("-n", "--num", type=int, default=None, help="The maximum number of documents to process from the start index.")

    # --- Generation Parameter Arguments ---
    parser.add_argument("-mt","--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate.")
    parser.add_argument("-nb","--num_beams", type=int, default=4, help="Number of beams for beam search.")
    parser.add_argument("-t","--temperature", type=float, default=0.7, help="The value used to module the next token probabilities.")
    parser.add_argument("-ds","--do_sample", action='store_true', help="Whether or not to use sampling; use greedy decoding otherwise.")

    args = parser.parse_args()

    input_file = args.input_file
    model_name = args.model_name
    output_folder = args.output_folder
    output_file = args.output_file
    num = args.num
    start = args.start
    end = args.end
    max_new_tokens = args.max_new_tokens
    num_beams = args.num_beams
    temperature = args.temperature
    do_sample = args.do_sample
    
    if args.start is not None and args.end is not None and args.start >= args.end:
        print("--start index must be less than --end index.")
        exit()

    try:
        with open('prompt_config.json', 'r', encoding='utf-8') as file:
            prompt_config = json.load(file)
    except FileNotFoundError:
        print("'prompt_config.json' file not found.")
        exit()
    except json.JSONDecodeError:
        print("Error decoding JSON from 'prompt_config.json'. Please check the file format.")
        exit()

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            pass 
    except FileNotFoundError:
        print(f"FATAL: Input file not found at '{args.input_file}'")
        exit()
   
    tokenizer, model = load_model_and_tokenizer(model_name)
    """ subset = load_malayalam_dataset(input_dataset, input_data_dir, start, end, num) """
    generate_and_save_queries(        tokenizer,
        model, input_file, output_folder, output_file, prompt_config,
        max_new_tokens, num_beams, temperature, do_sample, start, end, num
    )

if __name__ == "__main__":
    main_func()