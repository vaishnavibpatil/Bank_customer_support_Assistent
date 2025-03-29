from datasets import load_dataset
from transformers import BertTokenizer

# Load dataset from Hugging Face
dataset = load_dataset("legacy-datasets/banking77")

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Function to tokenize text
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=64)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Save tokenized dataset
tokenized_datasets.save_to_disk("tokenized_data")

print("Dataset preprocessing completed! ✅")
