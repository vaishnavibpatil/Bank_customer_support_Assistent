import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the trained model
model_path = "banking_intent_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Intent Mapping
intent_labels = {
    0: "Activate Credit Card",
    1: "Open New Account",
    2: "Close Account",
    3: "Update Account Info",
    4: "Minimum Balance Inquiry",
    5: "Apply for Credit Card",
    6: "Report Fraud/Hacked Account",
    7: "Activate Credit Card",
    8: "Increase debit Limit",
    9: "Block Lost Card",
    10: "Apply for a Loan",
    # Add the remaining intents...
}

def predict_intent(user_query):
    """Predict the intent of a given banking query"""
    inputs = tokenizer(user_query, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_id = torch.argmax(outputs.logits, dim=1).item()
    predicted_intent = intent_labels.get(predicted_id, "Unknown Intent")  # Convert ID to label
    
    return predicted_intent
