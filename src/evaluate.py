"""
Evaluates a trained model on the validation set.

Usage:
    python src/evaluate.py --data-dir data --checkpoint outputs/best_model.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from gradcam_utils import GradCAM, overlay_heatmap


def build_model(num_classes: int):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default="outputs/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--num-gradcam-examples", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = checkpoint["classes"]

    model = build_model(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_dir = Path(args.data_dir) / "val"
    val_ds = datasets.ImageFolder(val_dir, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    report = classification_report(all_labels, all_preds, target_names=classes, digits=3)
    print(report)
    with open(Path(args.output_dir) / "classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(max(6, len(classes) * 0.7), max(5, len(classes) * 0.6)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {args.output_dir}/confusion_matrix.png")

    misclassified_idx = np.where(all_preds != all_labels)[0]
    correct_idx = np.where(all_preds == all_labels)[0]
    n = args.num_gradcam_examples
    chosen_idx = list(misclassified_idx[: n // 2]) + list(correct_idx[: n - len(misclassified_idx[: n // 2])])
    chosen_idx = chosen_idx[:n]

    target_layer = model.layer4[-1]
    cam = GradCAM(model, target_layer)

    fig, axes = plt.subplots(1, len(chosen_idx), figsize=(4 * len(chosen_idx), 4))
    if len(chosen_idx) == 1:
        axes = [axes]

    for ax, idx in zip(axes, chosen_idx):
        img_tensor, true_label = val_ds[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)
        heatmap, pred_class = cam.generate(input_tensor)
        overlay = overlay_heatmap(img_tensor, heatmap, mean, std)

        ax.imshow(overlay)
        correct = pred_class == true_label
        ax.set_title(
            f"true: {classes[true_label]}\npred: {classes[pred_class]}",
            color="green" if correct else "red", fontsize=9,
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / "gradcam_examples.png", dpi=150)
    plt.close()
    print(f"Saved Grad-CAM examples to {args.output_dir}/gradcam_examples.png")


if __name__ == "__main__":
    main()
