"""Local-only upload demo for a research image-classification checkpoint.

It intentionally does not save uploads and never presents output as a blood type.
Do not deploy this service as a medical tool or connect it to clinical workflows.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from research_pipeline import LABELS, MEAN, STD, make_model

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
CHECKPOINT = Path(os.getenv("MODEL_CHECKPOINT", "models/model.pt"))
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

app = FastAPI(title="Fingerprint Study Lab — Local Research Demo")
_model = None
_held_out_accuracy = None


def load_model():
    global _model, _held_out_accuracy
    if _model is not None:
        return _model, _held_out_accuracy
    if not CHECKPOINT.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"No local checkpoint at {CHECKPOINT}. Train one first, then set MODEL_CHECKPOINT.",
        )
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if tuple(checkpoint.get("labels", ())) != LABELS:
        raise HTTPException(status_code=500, detail="Checkpoint label schema does not match this research demo.")
    model = make_model()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    _model = model
    _held_out_accuracy = checkpoint.get("held_out_test_accuracy")
    return _model, _held_out_accuracy


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Local Research Upload Demo</title><style>body{max-width:760px;margin:40px auto;padding:0 18px;background:#071421;color:#eff7ff;font:16px/1.5 system-ui}main{background:#0d2235;border:1px solid #23445d;border-radius:14px;padding:28px}h1{line-height:1.1}.warn{background:#37281f;border:1px solid #885843;padding:15px;border-radius:9px;color:#f7dbc2}input,button{font:inherit;margin:10px 0}button{background:#64e3d0;color:#062019;border:0;border-radius:8px;padding:10px 14px;font-weight:700}pre{white-space:pre-wrap;background:#06121d;border:1px solid #23445d;padding:14px;border-radius:9px;min-height:90px}</style></head><body><main><p>LOCAL RESEARCH DEMO · NOT A MEDICAL DEVICE</p><h1>Experimental image-classification upload</h1><div class='warn'><strong>Do not use this to determine blood group.</strong> Valid blood typing requires a blood sample and validated laboratory testing. Do not upload someone else's fingerprint or use any output for donation, transfusion, pregnancy, diagnosis, or treatment.</div><p>This server does not save selected images. It returns an experimental dataset-class score only after a local research checkpoint has been trained.</p><input id='file' type='file' accept='image/png,image/jpeg,image/webp'><br><button id='go'>Run research demo</button><pre id='result'>Waiting for a local research image.</pre></main><script>const result=document.getElementById('result');document.getElementById('go').onclick=async()=>{const file=document.getElementById('file').files[0];if(!file){result.textContent='Choose an image first.';return}result.textContent='Running local research checkpoint…';const form=new FormData();form.append('image',file);try{const response=await fetch('/api/research-score',{method:'POST',body:form});const body=await response.json();if(!response.ok)throw new Error(body.detail||'Request failed');const accuracy=body.held_out_test_accuracy===null?'not recorded':`${(body.held_out_test_accuracy*100).toFixed(2)}%`;result.textContent=`EXPERIMENTAL DATASET CLASS: ${body.experimental_dataset_class}\nMODEL SCORE: ${body.model_score}\nHELD-OUT TEST ACCURACY: ${accuracy}\n\n${body.medical_warning}`}catch(error){result.textContent=`No result: ${error.message}`}};</script></body></html>"""


@app.post("/api/research-score")
async def research_score(image: UploadFile = File(...)) -> dict[str, str | float]:
    if image.content_type not in {"image/png", "image/jpeg", "image/webp"}:
        raise HTTPException(status_code=415, detail="Use a PNG, JPEG, or WebP image.")
    raw = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than the 5 MB local-demo limit.")
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            tensor = TRANSFORM(opened.convert("RGB")).unsqueeze(0)
    except UnidentifiedImageError as error:
        raise HTTPException(status_code=400, detail="The selected file is not a readable image.") from error
    model, held_out_accuracy = load_model()
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        score, class_index = probabilities.max(dim=0)
    return {
        "experimental_dataset_class": LABELS[class_index.item()],
        "model_score": round(float(score.item()), 4),
        "held_out_test_accuracy": None if held_out_accuracy is None else round(float(held_out_accuracy), 4),
        "medical_warning": "Research-only score. This is not a blood-group result and must not be used for medical or donor decisions.",
    }
