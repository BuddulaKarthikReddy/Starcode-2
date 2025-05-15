import datasets
import os
from tqdm import tqdm
import logging
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize OpenAI client for vLLM server
try:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "dummy_key"),
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    )
    logging.info("Successfully initialized vLLM client at 02:29 PM EAT on Thursday, May 15, 2025")
except Exception as e:
    logging.error(f"Error initializing vLLM client: {e}")
    raise

# Load the instructions dataset from Step 2
logging.info("Loading instructions dataset from Step 2...")
try:
    instructions_dataset = datasets.load_dataset("json", data_files="../datasets/seeds_with_instructions.jsonl")['train']
    logging.info(f"Loaded instructions dataset with {len(instructions_dataset)} examples")
except Exception as e:
    logging.error(f"Error loading instructions dataset: {e}")
    raise

# I → R (Generate responses from instructions using vLLM)
logging.info("Generating responses from instructions using vLLM...")
responses = []
for i, example in enumerate(tqdm(instructions_dataset, desc="Generating responses", total=len(instructions_dataset))):
    instruction = example['instruction']
    try:
        # Prompt vLLM to generate a response (Java code)
        response = client.chat.completions.create(
            model="bigcode/starcoder2-15b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates Java code from natural language instructions."},
                {"role": "user", "content": f"Given the following instruction, write the corresponding Java code:\n\n{instruction}\n\nJava Code:"}
            ],
            max_tokens=200,
            temperature=0.0
        )
        generated_code = response.choices[0].message.content.strip()
    except Exception as e:
        logging.warning(f"Error generating response for instruction {i}: {e}")
        generated_code = example['seed']  # Fallback to original seed if vLLM fails
    
    responses.append({
        'instruction': instruction,
        'response': generated_code
    })
    if i % 100 == 0:  # Save every 100 examples
        temp_dataset = datasets.Dataset.from_list(responses)
        temp_dataset.save_to_disk(f"../datasets/instruction_response_pairs_{i}")
        logging.info(f"Saved temporary dataset at iteration {i}")

# Save the final dataset with instruction-response pairs
final_dataset = datasets.Dataset.from_list(responses)
final_dataset.save_to_disk("../datasets/instruction_response_pairs")
final_dataset.to_json("../datasets/instruction_response_pairs.jsonl", orient="records", lines=True)
logging.info(f"Saved final dataset with {len(final_dataset)} instruction-response pairs to ../datasets/instruction_response_pairs.jsonl")