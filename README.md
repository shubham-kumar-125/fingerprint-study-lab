# Fingerprint Study Lab

> **Research and education only — not a medical device.** A fingerprint cannot be used to establish a person's blood type for diagnosis, donation, transfusion, pregnancy care, or any medical decision. Blood type must be established with validated blood testing by qualified services.

This project is a reproducible image-classification research scaffold inspired by the public reference repository's dataset layout and CNN comparison. It deliberately improves the research workflow with participant-aware data splits, held-out evaluation, calibration reporting, a model card, and explicit non-clinical safeguards.

## What is included

- `research_pipeline.py` — train, evaluate, and inspect a PyTorch ResNet-18 baseline
- `model_card.md` — intended use, limits, evaluation requirements, and safety boundaries
- `data/` — the place for an approved, de-identified research dataset (not included)
- `dist/index.html` — a static public research site that never accepts fingerprint uploads or presents a blood-typing result

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset contract

Do **not** use a dataset unless you have permission, consent, a legitimate research purpose, and an approved privacy plan. Fingerprints are biometric data.

Create `data/metadata.csv` with these exact columns:

```csv
relative_path,label,participant_id
images/p001_left_index.png,A+,p001
images/p002_right_thumb.png,O-,p002
```

The label must be one of `A+`, `A-`, `B+`, `B-`, `AB+`, `AB-`, `O+`, or `O-`. Images from one participant must share one `participant_id`; the pipeline keeps a participant in only one split to avoid overly optimistic results caused by fingerprint leakage.

## Train and evaluate

```bash
python research_pipeline.py train --data-dir data --output-dir outputs/run-001 --epochs 12
python research_pipeline.py predict --checkpoint outputs/run-001/model.pt --image data/images/p001_left_index.png
```

`predict` prints an experimental class score with a non-clinical warning. It is not a blood-type result and must never be used as one.

## Run the local upload demo

After training a checkpoint, point the local-only demo at it. It processes a chosen image in memory and does not save uploads.

```bash
MODEL_CHECKPOINT=outputs/run-001/model.pt uvicorn local_app:app --reload
```

Open `http://127.0.0.1:8000`. The page displays only an **experimental dataset-class score** and a permanent warning. It must not be presented as a blood-group test or used with real people.

## Publish the static research site

The public research site is live at [Fingerprint Study Lab](https://shubham-kumar-125.github.io/fingerprint-study-lab/). The `dist/` folder is a self-contained static site that can also be opened locally as `dist/index.html`.
