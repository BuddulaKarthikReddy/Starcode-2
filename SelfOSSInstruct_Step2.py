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

# Load the seeds dataset
logging.info("Loading seeds dataset from Step 1...")
try:
    seeds_dataset = datasets.load_dataset("json", data_files="../datasets/seeds.jsonl")['train']
    logging.info(f"Loaded seeds dataset with {len(seeds_dataset)} examples")
except Exception as e:
    logging.error(f"Error loading seeds dataset: {e}")
    raise

# Sub-step 1: S → C (Generate comments from seeds using vLLM)
logging.info("Generating comments from seeds using vLLM...")
comments = []
for i, example in enumerate(tqdm(seeds_dataset, desc="Generating comments", total=len(seeds_dataset))):
    seed = example['seed']
    try:
        # Prompt vLLM to generate a comment
        response = client.chat.completions.create(
            model="bigcode/starcoder2-15b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates comments for Java code."},
                {"role": "user", "content": f"Generate a brief comment describing what the following Java code does:\n\n{seed}\n\nComment:"}
            ],
            max_tokens=100,
            temperature=0.0
        )
        comment = response.choices[0].message.content.strip()
    except Exception as e:
        logging.warning(f"Error generating comment for seed {example['id']}: {e}")
        comment = f"// Unable to generate comment for this method."
    
    example['comment'] = comment
    comments.append(example)
    if i % 100 == 0:  # Save every 100 examples
        temp_dataset = datasets.Dataset.from_list(comments)
        temp_dataset.save_to_disk(f"../datasets/seeds_with_comments_{i}")
        logging.info(f"Saved temporary dataset at iteration {i}")

# Save the dataset with comments
comments_dataset = datasets.Dataset.from_list(comments)
comments_dataset.save_to_disk("../datasets/seeds_with_comments")
logging.info(f"Saved dataset with comments: {len(comments_dataset)} examples")

# Sub-step 2: C → I (Generate instructions from comments using vLLM)
logging.info("Generating instructions from comments using vLLM...")
instructions = []
for i, example in enumerate(tqdm(comments_dataset, desc="Generating instructions", total=len(comments_dataset))):
    comment = example['comment']
    try:
        # Prompt vLLM to generate an instruction
        response = client.chat.completions.create(
            model="bigcode/starcoder2-15b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that converts comments into natural language instructions for writing Java code."},
                {"role": "user", "content": f"Convert the following comment into a natural language instruction for writing Java code:\n\n{comment}\n\nInstruction:"}
            ],
            max_tokens=100,
            temperature=0.0
        )
        instruction = response.choices[0].message.content.strip()
    except Exception as e:
        logging.warning(f"Error generating instruction for comment {example['id']}: {e}")
        instruction = f"Write a Java method based on the comment: {comment}"
    
    example['instruction'] = instruction
    instructions.append(example)
    if i % 100 == 0:  # Save every 100 examples
        temp_dataset = datasets.Dataset.from_list(instructions)
        temp_dataset.save_to_disk(f"../datasets/seeds_with_instructions_{i}")
        logging.info(f"Saved temporary dataset at iteration {i}")

# Save the dataset with instructions
instructions_dataset = datasets.Dataset.from_list(instructions)
instructions_dataset.save_to_disk("../datasets/seeds_with_instructions")
instructions_dataset.to_json("../datasets/seeds_with_instructions.jsonl", orient="records", lines=True)
logging.info(f"Saved dataset with instructions: {len(instructions_dataset)} examples to ../datasets/seeds_with_instructions.jsonl")