"""
Internet Comments Aggression Filtering - Gradio demonstration.

Expected structure:
project/
  code/demo_gradio.py
  distilbert_model/
    config.json
    model.safetensors (or pytorch_model.bin)
    tokenizer files...

Run from the project root:
  pip install -r requirements.txt
  python code/demo_gradio.py

The model was trained on the English Jigsaw Toxic Comment dataset.
It is not validated for Lithuanian-language comments.
"""

from pathlib import Path
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "distilbert_model"
MAX_LENGTH = 128

if not (MODEL_DIR / "config.json").exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_DIR}. "
        "Download/extract the trained distilbert_model folder there first."
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.to(device)
model.eval()


def classify_comment(comment: str):
    if not comment or not comment.strip():
        return {"Įvesk komentarą": 1.0}

    inputs = tokenizer(
        comment.strip(),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()

    return {
        "Normalus komentaras": float(probabilities[0]),
        "Toksiškas komentaras": float(probabilities[1]),
    }


demo = gr.Interface(
    fn=classify_comment,
    inputs=gr.Textbox(
        lines=4,
        label="Įvesk komentarą",
        placeholder="Write a comment in English...",
    ),
    outputs=gr.Label(num_top_classes=2, label="Modelio rezultatas"),
    title="Internet Comments Aggression Filtering",
    description=(
        "DistilBERT modelis analizuoja anglišką komentarą ir įvertina "
        "jo toksiškumo tikimybę. Modelis mokytas su Jigsaw anglų kalbos duomenimis."
    ),
    examples=[
        ["I disagree with your opinion, but I respect your point of view."],
        ["You are an idiot and nobody likes you."],
    ],
)

if __name__ == "__main__":
    demo.launch()
