import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm

# Control whether to train or load the model
TRAIN_MODEL = True  # Set to True to train; False to load and evaluate/predict

# Load train and test datasets
train_data = pd.read_csv("./Dataset/yelp_review_fine-grained_5_classes_csv/train.csv")
test_data = pd.read_csv("./Dataset/yelp_review_fine-grained_5_classes_csv/test.csv")

# Rename columns for clarity
train_data.rename(columns={"class_index": "rating", "review_text": "review"}, inplace=True)
test_data.rename(columns={"class_index": "rating", "review_text": "review"}, inplace=True)

# Stratified sampling for subsets
train_subset, _ = train_test_split(train_data, train_size=10000, stratify=train_data['rating'], random_state=42)
test_subset, _ = train_test_split(test_data, train_size=2000, stratify=test_data['rating'], random_state=42)

# Scale ratings and reset index
train_subset['rating'] = (train_subset['rating'] - 1) / 4
test_subset['rating'] = (test_subset['rating'] - 1) / 4
train_subset.reset_index(drop=True, inplace=True)
test_subset.reset_index(drop=True, inplace=True)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize data function
def tokenize_data(reviews, ratings):
    return tokenizer(
        list(reviews),
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    ), torch.tensor(ratings.values, dtype=torch.float)

# Tokenize training and testing data
train_inputs, train_labels = tokenize_data(train_subset['review'], train_subset['rating'])
test_inputs, test_labels = tokenize_data(test_subset['review'], test_subset['rating'])

# Create TensorDataset and DataLoaders
train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if TRAIN_MODEL:
    # Load BERT model for regression
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=1  # Regression output
    ).to(device)

    # Define optimizer and loss
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = torch.nn.MSELoss()

    # Training loop
    epochs = 3
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader)}")

    # Save the trained model
    torch.save(model.state_dict(), "trained_model_10_000.pth")
    print("Model training complete and saved to 'trained_model.pth'.")
else:
    # Load the trained model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=1
    ).to(device)
    model.load_state_dict(torch.load("trained_model.pth"))
    model.eval()
    print("Loaded trained model from 'trained_model.pth'.")

# Evaluate model on the test set
model.eval()
loss_fn = torch.nn.MSELoss()
test_loss = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = outputs.logits.squeeze()  # Squeeze to remove extra dimension
        test_loss += loss_fn(predictions, labels).item()

print(f"Test Loss (MSE): {test_loss / len(test_loader)}")

# Prediction function
def predict_sentiment(sentence):
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        sentiment_score = outputs.logits.squeeze().item() * 100  # Direct scaling
    return sentiment_score

# Example usage of the prediction function
test_sentence = "The product is amazing!"
print(f"Sentiment Score: {predict_sentiment(test_sentence):.2f}%")
print(f"Sentiment Score: {predict_sentiment("I hate being alone in the dark."):.2f}%")
print(f"Sentiment Score: {predict_sentiment("asfsdfsfkhfsd  sdfhkshfjkshfskj sjdkfhskfjh fgakjfg"):.2f}%")
print(f"Sentiment Score: {predict_sentiment("I hate other people I want to murder them all"):.2f}%")
