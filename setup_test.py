import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import csv
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

def load_malayalam_dataset(input_dataset, input_data_dir, num):
    print(f"Attempting to load dataset {input_dataset} with data_dir='{input_data_dir}'")
    try:
        dataset = load_dataset(input_dataset, data_dir=input_data_dir, split='train')
        if num is None:
            subset = dataset
            print(f"Successfully Loaded ALL {len(subset)} documents \n")
        else:
            num_select = min(num, len(dataset))
            subset = dataset.select(range(num_select))
            print(f"Successfully Loaded {len(subset)} documents \n")
        return subset
    except Exception as e:
        print(f"Failed to load documents: {e}")
        exit()

def format_prompt(domain, prompt_config, document):
    if domain not in prompt_config:
        print(f"Domain '{domain}' not found in prompt_config.json. Available domains: {list(prompt_config.keys())}")
        exit()

    few_shots = prompt_config[domain]
    ex = ""
    for example in few_shots:
        ex += f"DOCUMENT:\n{example["document"]} \n USER QUERY: {example["query"]}\n\n"

    prompt = (
        f"Your task is to create a single, natural question in the Malayalam language that a user in the domain {domain} would ask to find the given document. The generated query must be in Malayalam.\n\n"
        "---EXAMPLES---\n"
        f"{ex}"
        "---YOUR TASK---\n\n"
        f"DOCUMENT:\n {document}"
        "USER QUERY: "
    )

    return prompt

def generate_and_save_queries(tokenizer, model, subset, output_folder, output_file, prompt_config, domain, max_new_tokens, num_beams, temperature, do_sample):
    os.makedirs(output_folder, exist_ok=True)
    output_csv_path = os.path.join(output_folder, f'{output_file}.csv')

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['document_text', 'generated_query'])

        print(f"Starting query generation and saving results to {output_csv_path}... \n")

        for i, item in enumerate(subset):
            document_text = item['text']
            print(f"\nProcessing document {i+1}/{len(subset)}...")
            print(f"Document:\t{document_text}")
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
            writer.writerow([document_text, generated_query])

    print(f"\nProcessing complete. Results saved to {output_csv_path}")


def main_func():
    parser = argparse.ArgumentParser(description="Generate synthetic queries for Malayalam documents.")
    parser.add_argument(
        "-d", "--input_dataset",
        type=str,
        default="ai4bharat/sangraha",
        help="Dataset to be loaded. Eg: 'ai4bharat/sangraha' "
    )
    parser.add_argument(
        "-i", "--input_data_dir",
        type=str,
        default="verified/mal",
        help="Subdirectory within 'ai4bharat/sangraha' dataset to load, Eg: 'verified/mal'."
    )
    parser.add_argument(
        "-m", "--model_name",
        type=str,
        default="google/gemma-3-1b-it",
        help="Hugging Face model ID to use for query generation. Eg: 'google/gemma-3-1b-it'."
    )
    parser.add_argument(
        "-o", "--output_folder",
        type=str,
        default="results",
        help="Folder where the synthetic queries will be saved. Will be created if it doesn't exist."
    )
    parser.add_argument(
        "-f", "--output_file",
        type=str,
        default="generated_queries",
        help="Base name for the output CSV file (e.g., 'generated_queries' will create 'generated_queries.csv')."
    )
    parser.add_argument(
        "-n", "--num_queries",
        type=int,
        default=None,  
        help="Number of synthetic queries to generate from the dataset. Set to 0 or omit to process all documents."
    )
    parser.add_argument(
        "-c", "--domain",
        type=str,
        default="politics",
        help="""Domains: politics, government, health, education, sports, entertainment, technology, science,
                finance, environment, law, crime, culture, literature, weather, travel, news, religion, 
                spirituality, business, industrial, food and drink, vehicles, pets and animals, 
                home and garden, computers and electronics, style and fashion, real estate, shopping, 
                hobbies and interests, arts, history, music"""
    )

    args = parser.parse_args()
    input_dataset = args.input_dataset
    model_id = args.model_name
    input_data_dir = args.input_data_dir
    output_folder = args.output_folder
    output_file = args.output_file
    num = args.num_queries
    domain = args.domain

    max_new_tokens = 50
    num_beams = 4
    temperature = 0.7
    do_sample = False

    try:
        with open('prompt_config.json', 'r', encoding='utf-8') as file:
            prompt_config = json.load(file)
    except FileNotFoundError:
        print("prompt_config.json file not found")
        exit()
    except json.JSONDecodeError:
        print("Error decoding json from file")
        exit()

    if domain not in prompt_config:
        print(f"Domain {domain} not found in prompt_config.json. Available domains: {list(prompt_config.keys())}")
        exit()

    tokenizer, model = load_model_and_tokenizer(model_id)
    subset = load_malayalam_dataset(input_dataset, input_data_dir, num)
    generate_and_save_queries(tokenizer, model, subset, output_folder, output_file, prompt_config, domain,
                              max_new_tokens, num_beams, temperature, do_sample)

if __name__ == "__main__":
    main_func()