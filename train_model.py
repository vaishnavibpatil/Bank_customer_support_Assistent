import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# Load tokenized dataset
dataset = load_from_disk("tokenized_data")

# Load the model (BERT for classification)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=77).to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="banking_intent_model",
    eval_strategy="epoch",  # Updated to avoid deprecation warning
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16 if torch.cuda.is_available() else 8,  # Use larger batch size if GPU is available
    per_device_eval_batch_size=16 if torch.cuda.is_available() else 8,
    logging_dir="./logs",
    logging_steps=50,  # Log more frequently for better monitoring
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none"  # Prevents sending logs to WandB/Hugging Face (optional)
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

# Train the model
print("🚀 Starting training...")
trainer.train()

# Save the fine-tuned model
model.save_pretrained("banking_intent_model")

print("✅ Model training completed successfully!")
