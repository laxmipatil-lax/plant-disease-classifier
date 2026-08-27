"""
Streamlit demo: upload an image, see the predicted class, confidence scores,
and a Grad-CAM overlay showing what the model focused on.

Run with:
    streamlit run app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models

sys.path.append(str(Path(__file__).parent / "src"))
from gradcam_utils import GradCAM, overlay_heatmap  # noqa: E402

MODEL_PATH = "outputs/best_model.pt"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    classes = checkpoint["classes"]

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, classes


def preprocess(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return transform(image)


st.set_page_config(page_title="Image Classifier Demo", page_icon="🖼️", layout="centered")
st.title("🖼️ Image Classifier")
st.write("Upload an image to see the model's prediction, confidence, and what it's paying attention to.")

if not Path(MODEL_PATH).exists():
    st.error(
        f"No trained model found at `{MODEL_PATH}`. "
        "Run `python src/train.py --data-dir data` first to train and save a model."
    )
    st.stop()

model, classes = load_model()
target_layer = model.layer4[-1]
cam = GradCAM(model, target_layer)

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_tensor = preprocess(image)
    input_tensor = img_tensor.unsqueeze(0)

    with st.spinner("Running inference..."):
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1).squeeze().numpy()

        # Grad-CAM needs gradients, so run it separately from the no_grad inference above
        heatmap, pred_class = cam.generate(input_tensor)
        overlay = overlay_heatmap(img_tensor, heatmap, MEAN, STD)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Grad-CAM")
        st.image(overlay, use_container_width=True, clamp=True)

    st.subheader(f"Prediction: **{classes[pred_class]}**")
    st.write(f"Confidence: **{probs[pred_class] * 100:.1f}%**")

    st.subheader("All class probabilities")
    sorted_idx = np.argsort(probs)[::-1]
    for i in sorted_idx:
        st.write(f"{classes[i]}: {probs[i] * 100:.1f}%")
        st.progress(float(probs[i]))
