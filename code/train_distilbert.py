
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "data/train.csv"
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 1

class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[idx]))
        }

def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["comment_text"])

    train_df, val_df = train_test_split(
        df[["comment_text", "toxic"]],
        test_size=0.2,
        random_state=42,
        stratify=df["toxic"]
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    train_dataset = CommentDataset(
        train_df["comment_text"], train_df["toxic"], tokenizer
    )
    val_dataset = CommentDataset(
        val_df["comment_text"], val_df["toxic"], tokenizer
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=16)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()

        print(f"Epoch {epoch + 1}, loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    predictions, labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            labels.extend(batch["labels"].tolist())
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions.extend(
                torch.argmax(outputs.logits, dim=1).cpu().tolist()
            )

    print("Accuracy:", accuracy_score(labels, predictions))
    print("Precision:", precision_score(labels, predictions))
    print("Recall:", recall_score(labels, predictions))
    print("F1-score:", f1_score(labels, predictions))
    print(classification_report(labels, predictions))
    print("Confusion matrix:")
    print(confusion_matrix(labels, predictions))

    model.save_pretrained("distilbert_model")
    tokenizer.save_pretrained("distilbert_model")

if __name__ == "__main__":
    main()
