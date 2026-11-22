
import re
import pandas as pd


def clean_text(text):
    """Cleans a comment before AI classification."""

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_dataset(input_file, output_file):
    df = pd.read_csv(input_file)

    if "comment_text" not in df.columns:
        raise ValueError("Dataset must contain comment_text column")

    df = df.dropna(subset=["comment_text"])
    df = df.drop_duplicates(subset=["comment_text"])

    df["cleaned_text"] = df["comment_text"].apply(clean_text)

    df.to_csv(output_file, index=False)

    print("Data preprocessing completed!")
    print(f"Processed comments: {len(df)}")


if __name__ == "__main__":
    preprocess_dataset(
        "data/train.csv",
        "data/cleaned_comments.csv"
    )
