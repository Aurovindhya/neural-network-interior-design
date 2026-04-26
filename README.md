# interior-style-nn

A neural network that classifies interior design styles from room images using transfer learning on EfficientNet, served via a FastAPI inference endpoint with Langfuse observability.

## What It Does

Upload a room photo → get a predicted design style + confidence scores + complementary color palette recommendations.

**Supported styles:**
- Mid-Century Modern
- Scandinavian
- Industrial
- Bohemian
- Minimalist

## Architecture

```
Room Image
    │
    ▼
EfficientNet-B0 (pretrained ImageNet backbone)
    │
    ▼
Custom classifier head (512 → 5 classes)
    │
    ▼
Style prediction + confidence scores
    │
    ▼
Rule-based palette recommender
    │
    ▼
FastAPI response + Langfuse trace
```

Transfer learning strategy: freeze the backbone for the first 5 epochs, then unfreeze and fine-tune the top layers at a lower learning rate.

## Project Structure

```
interior-style-nn/
├── data/
│   └── scripts/
│       └── download_dataset.py   # Downloads ~150 curated images via URLs
├── model/
│   ├── model.py                  # EfficientNet model definition
│   ├── dataset.py                # PyTorch Dataset + transforms
│   ├── train.py                  # Training script (CLI)
│   └── evaluate.py               # Evaluation + confusion matrix
├── api/
│   ├── main.py                   # FastAPI app
│   ├── predictor.py              # Inference logic
│   └── schemas.py                # Pydantic request/response models
├── langfuse/
│   └── tracing.py                # Langfuse integration
├── notebooks/
│   └── InteriorStyleNN_Colab.ipynb  # End-to-end Colab notebook
├── requirements.txt
├── .env.example
└── README.md
```

## Quickstart (Inference Only)

If you just want to run the API with the pretrained weights:

```bash
git clone https://github.com/yourusername/interior-style-nn
cd interior-style-nn
pip install -r requirements.txt
cp .env.example .env   # add your Langfuse keys
uvicorn api.main:app --reload
```

Then POST an image:
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@your_room.jpg"
```

## Training (Google Colab recommended)

Open the notebook in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/interior-style-nn/blob/main/notebooks/InteriorStyleNN_Colab.ipynb)

Or run locally with a GPU:

```bash
# 1. Download dataset
python data/scripts/download_dataset.py

# 2. Train
python model/train.py --epochs 15 --batch-size 32 --output weights/best_model.pth

# 3. Evaluate
python model/evaluate.py --weights weights/best_model.pth
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Upload image, get style prediction |
| `GET`  | `/styles` | List all supported styles |
| `GET`  | `/health` | Health check |

### Example Response

```json
{
  "style": "Mid-Century Modern",
  "confidence": 0.87,
  "all_scores": {
    "Mid-Century Modern": 0.87,
    "Scandinavian": 0.08,
    "Industrial": 0.03,
    "Bohemian": 0.01,
    "Minimalist": 0.01
  },
  "palette": {
    "primary": ["#8B4513", "#D2691E", "#F5DEB3"],
    "accent": ["#2F4F4F", "#708090"],
    "description": "Warm walnut tones with slate accents — characteristic of mid-century modern interiors."
  },
  "trace_id": "lf-abc123"
}
```

## Langfuse Observability

Every inference call is traced with:
- Input image metadata (size, format)
- Model prediction + confidence
- Latency
- Any errors

Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env` to enable. Tracing degrades gracefully if keys are absent.

## Dataset

~150 images across 5 classes (30 per style), sourced from open-licensed interior design photography. Run `python data/scripts/download_dataset.py` to fetch them.

For more data, the training script accepts any folder with `class_name/image.jpg` structure — drop in additional images and retrain.

## Model Performance

On the included 150-image dataset (80/20 train/val split):

| Metric | Value |
|--------|-------|
| Val Accuracy | ~82% |
| Val Loss | ~0.51 |
| Inference time | ~45ms (CPU) |

With a larger dataset (500+ images per class), expect 90%+ accuracy.

## Tech Stack

- **PyTorch** + **torchvision** — model and training
- **EfficientNet-B0** — pretrained backbone via `timm`
- **FastAPI** — inference API
- **Pillow** — image preprocessing
- **Langfuse** — inference tracing and evaluation
- **Google Colab** — recommended training environment
