# streamlit_app.py — SolarGuard (Task-1: Classification)

from pathlib import Path
import io
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    resnet50, ResNet50_Weights
)

# --> paths 
THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parents[1]          # .../solar_guard
ARTIFACTS    = PROJECT_ROOT / "artifacts"
REPORTS      = PROJECT_ROOT / "reports" / "figures"
CKPT_PATH    = ARTIFACTS / "best_classifier.pt"      
CM_PATH      = REPORTS / "confusion_matrix_test.png" 

# --> simple model factory 
def make_model(name: str, num_classes: int):
    name = name.lower()
    if name == "mobilenet_v3":
        m = mobilenet_v3_small(weights=None)
        in_feats = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_feats, num_classes)
        return m
    if name == "efficientnet_b0":
        m = efficientnet_b0(weights=None)
        in_feats = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feats, num_classes)
        return m
    if name == "resnet50":
        m = resnet50(weights=None)
        in_feats = m.fc.in_features
        m.fc = nn.Linear(in_feats, num_classes)
        return m
    raise ValueError("unknown model name")

# --> loading checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not CKPT_PATH.exists():
    raise FileNotFoundError(f"Missing: {CKPT_PATH}. Train the model in the notebook first.")

ckpt = torch.load(CKPT_PATH, map_location=device)
class_to_idx = ckpt["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes  = len(idx_to_class)
model_name   = ckpt.get("model_name", "mobilenet_v3")

model = make_model(model_name, num_classes)
model.load_state_dict(ckpt["model"])
model.eval().to(device)

# same eval transforms used in notebook
infer_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

def predict(img: Image.Image):
    """Return list of (class, prob) sorted by prob desc."""
    im = img.convert("RGB")
    t = infer_tfms(im).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
        probs = out.softmax(dim=1).squeeze(0).cpu().numpy()
    order = np.argsort(probs)[::-1]
    return [(idx_to_class[i], float(probs[i])) for i in order]

# --> UI 
st.set_page_config(page_title="SolarGuard — Classification", layout="centered")
st.title("SolarGuard — Solar Panel Classification")
st.caption("Classes: BirdDrop, Clean, Dusty, ElectricalDamage, PhysicalDamage, SnowCovered")

col1, col2 = st.columns([1,1])

with col1:
    file = st.file_uploader("Upload a panel image", type=["jpg","jpeg","png","bmp","webp"])
    if file:
        img = Image.open(io.BytesIO(file.read()))
        st.image(img, caption="Input", use_column_width=True)

        preds = predict(img)
        top_cls, top_p = preds[0]
        st.subheader(f"Prediction: {top_cls}")
        st.write(f"Confidence: **{top_p:.2%}**")

        # show top-k as a small bar chart
        df = pd.DataFrame({"class":[c for c,_ in preds],
                           "probability":[p for _,p in preds]})
        st.bar_chart(df.set_index("class"))

with col2:
    st.markdown("**Confusion Matrix (Test)**")
    if CM_PATH.exists():
        st.image(str(CM_PATH), use_column_width=True)
    else:
        st.info("Confusion matrix not found — run evaluation in the notebook to generate it.")

st.divider()
st.markdown(
    f"Model: **{model_name}** | Val Macro F1: **{ckpt.get('val_f1', -1.0):.4f}** | Device: **{device}**"
)
