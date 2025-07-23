import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import csv 
import argparse

os.environ['HF_HOME'] = 'D:/huggingface_cache'


model_id = "google/gemma-3-1b-it"

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
        dataset = load_dataset(input_dataset, data_dir=input_data_dir)
        subset = dataset['train'].select(range(num))
        print(f"Successfully Loaded {len(subset)} documents \n")
        return subset
    except Exception as e:
        print(f"Failed to load documents: {e}")
        exit()

def generate_and_save_queries(tokenizer, model, subset, output_folder, max_new_tokens, num_beams, temperature, do_sample):
    
    os.makedirs(output_folder, exist_ok=True)
    output_csv_path = os.path.join(output_folder, 'generated_queries.csv')

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['document_text', 'generated_query'])

        print(f"Starting query generation and saving results to {output_csv_path}... \n")

        doc_1 = "ദുബായ് : ടി20 ലോകകപ്പിൽ ബംഗ്ലാദേശിന്റെ ശേഷിക്കുന്ന മത്സരങ്ങളിൽ നിന്ന് ഷാക്കിബ് അൽ ഹസൻ പരിക്ക് കാരണം പുറത്തായി."
        query_1 = "ടി20 ലോകകപ്പിൽ നിന്ന് ഷാക്കിബ് അൽ ഹസൻ പുറത്താകാൻ കാരണമെന്ത്?"

        doc_2 = "കേരളത്തിൽ ഇന്ന് 10 പുതിയ കോവിഡ് കേസുകൾ റിപ്പോർട്ട് ചെയ്തു. ആരോഗ്യ വകുപ്പ് മന്ത്രി കെ.കെ. ശൈലജ ടീച്ചർ അറിയിച്ചതാണ് ഇക്കാര്യം."
        query_2 = "കേരളത്തിൽ ഇന്ന് എത്ര കോവിഡ് കേസുകൾ റിപ്പോർട്ട് ചെയ്തു?"

        for i, item in enumerate(subset):
            document_text = item['text']
            print(f"\nProcessing document {i+1}/{len(subset)}...")
            print(f"Document:\t{document_text}")
            prompt = (
                "Your task is to create a single, natural question in the Malayalam language that a user would ask to find the given document. The generated query must be in Malayalam. \n\n"
                "--- EXAMPLES ---\n"
                f"DOCUMENT: {doc_1} \n"
                f"USER QUERY: {query_1}\n\n"
                f"DOCUMENT: {doc_2} \n"
                f"USER QUERY: {query_2}\n\n"
                "--- TASK ---\n"
                f"DOCUMENT:\n{document_text}\n\n"
                f"USER QUERY:"
            )

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dataset",
        type=str,
        default="ai4bharat/sangraha",
        help="Dataset to be loaded. Eg: 'ai4bharat/sangraha' "
    )
    parser.add_argument(
        "--input_data_dir",
        type=str,
        default="verified/mal", 
        help="Subdirectory within 'ai4bharat/sangraha' dataset to load, Eg: 'verified/mal'."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-1b-it", 
        help="Hugging Face model ID to use for query generation. Eg: 'google/gemma-3-1b-it'."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="generated_queries", 
        help="Folder where the synthetic queries will be saved. Will be created if it doesn't exist."
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=100, 
        help="Number of synthetic queries to generate from the dataset."
    )
    
    args = parser.parse_args()
    input_dataset = args.input_dataset
    model_id = args.model_name
    input_data_dir = args.input_data_dir
    output_folder = args.output_folder
    num = args.num_queries

    max_new_tokens = 50
    num_beams = 4
    temperature = 0.0
    do_sample = False

    tokenizer, model = load_model_and_tokenizer(model_id)
    subset = load_malayalam_dataset(input_dataset, input_data_dir, num)
    generate_and_save_queries(tokenizer, model, subset, output_folder,
                              max_new_tokens, num_beams, temperature, do_sample)

if __name__ == "__main__":
    main_func()
