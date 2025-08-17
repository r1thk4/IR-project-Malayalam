import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import argparse
import json

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

def load_malayalam_dataset(input_dataset, input_data_dir, start_index, end_index, num):
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
        exit()

def format_prompt(domain, prompt_config, document_text):
    if domain not in prompt_config:
        print(f"Domain '{domain}' not found in prompt_config.json. Available domains: {list(prompt_config.keys())}")
        exit()

    few_shots = prompt_config[domain]
    ex = ""
    for example in few_shots:
        ex += f"DOCUMENT:\n{example['document']} \n USER QUERY: {example['query']}\n\n"

    prompt = (
        f"Your task is to create a single, natural question in the Malayalam language that a user in the domain {domain} would ask to find the given document. The generated query must be in Malayalam.\n\n"
        "---EXAMPLES---\n"
        f"{ex}"
        "---YOUR TASK---\n"
        f"DOCUMENT:\n{document_text}\n\n"
        "USER QUERY:"
    )
    return prompt

def generate_and_save_queries(tokenizer, model, subset, output_folder, output_file, prompt_config, domain, max_new_tokens, num_beams, temperature, do_sample):
    os.makedirs(output_folder, exist_ok=True)
    output_jsonl_path = os.path.join(output_folder, f'{output_file}.jsonl')

    with open(output_jsonl_path, 'w', newline='', encoding='utf-8') as jsonlfile:

        print(f"Starting query generation and saving results to {output_jsonl_path}... \n")
        
        for i, item in enumerate(subset):
            document_text = item['text']
            print(f"\nProcessing document {i+1}/{len(subset)}...")
            print(f"Document:\t{document_text}")
            if not document_text.strip():
                print("Skipping empty document.")
                continue
            prompt = format_prompt(domain, prompt_config, document_text)
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

            print(f"Generated Query:\t{generated_query}")
            record = {
                "document": document_text,
                "query": generated_query
            }
            jsonlfile.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"\nProcessing complete. Results saved to {output_jsonl_path}")


def main_func():
    parser = argparse.ArgumentParser(description="Generate synthetic queries for Malayalam documents.")
    
    parser.add_argument("-d", "--input_dataset", type=str, default="ai4bharat/sangraha", help="Dataset to be loaded.")
    parser.add_argument("-i", "--input_data_dir", type=str, default="verified/mal", help="Subdirectory within the dataset.")
    parser.add_argument("-m", "--model_name", type=str, default="google/gemma-2b-it", help="Hugging Face model ID.")
    parser.add_argument("-o", "--output_folder", type=str, default="results", help="Folder to save the output.")
    parser.add_argument("-f", "--output_file", type=str, default=None, help="Name for the output JSONL file. Defaults to 'queries_[domain].jsonl'.")
    parser.add_argument("-c", "--domain", type=str, default="politics", help="The domain to process (e.g., 'politics', 'sports').")
    parser.add_argument("-n", "--num", type=int, default=None, help="Number of synthetic queries to generate from the dataset. Set to 0 or omit to process all documents.")
    parser.add_argument("-s", "--start", type=int, default=None, help="The starting index of the documents to process.")
    parser.add_argument("-e", "--end", type=int, default=None, help="The ending index (exclusive) of the documents to process.")
    parser.add_argument("-mt", "--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate.")
    parser.add_argument("-nb", "--num_beams", type=int, default=4, help="Number of beams for beam search.")
    parser.add_argument("-t", "--temperature", type=float, default=0.7, help="The value used to module the next token probabilities.")
    parser.add_argument("-ds", "--do_sample", action='store_true', help="Whether or not to use sampling; use greedy decoding otherwise.")
    args = parser.parse_args()

    input_dataset = args.input_dataset
    model_name = args.model_name
    input_data_dir = args.input_data_dir
    domain = args.domain
    output_folder = args.output_folder
    output_file = args.output_file if args.output_file else f"queries_{domain}"
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

   
    tokenizer, model = load_model_and_tokenizer(model_name)
    subset = load_malayalam_dataset(input_dataset, input_data_dir, start, end, num)
    generate_and_save_queries(        tokenizer,
        model, subset, output_folder, output_file, prompt_config, domain,
        max_new_tokens, num_beams, temperature, do_sample
    )

if __name__ == "__main__":
    main_func()