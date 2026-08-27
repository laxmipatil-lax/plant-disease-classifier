# Plant Disease Classifier — Image Classification

Classifying plant leaf images into 25 crop disease and healthy categories, to help catch crop disease early using computer vision.

🔗 **Repo:** [this repo](https://github.com/laxmipatil-lax/plant-disease-classifier)

---

## The Problem

Crop diseases cause major yield losses worldwide, and early detection is one of the most effective ways to limit their spread. Manually inspecting every plant is slow and requires expertise many farmers don't have easy access to. This project explores whether a computer vision model can classify plant leaf images into disease categories accurately enough to be useful as an early-warning tool.

## The Dataset

- **Source:** [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Size:** ~35,000 leaf images across 25 classes (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean — disease and healthy variants)
- **Why this dataset:** It's a widely-used, well-labeled benchmark for plant disease classification, covering multiple crops and both diseased and healthy leaf conditions, making it realistic for a real-world early-detection tool.

## Approach

- **Model:** Transfer learning on `resnet18` (pretrained on ImageNet), fine-tuned on the target dataset
- **Preprocessing:** resize to 224x224, ImageNet mean/std normalization, augmentation (random flip, rotation, color jitter) applied on the training split only
- **Evaluation:** confusion matrix, per-class precision/recall/F1, and Grad-CAM visualizations to inspect which regions of each leaf the model focuses on

## Project Structure
plant-disease-classifier/
├── app.py # Streamlit demo app
├── src/
│ ├── train.py # Model training script (transfer learning on ResNet18)
│ ├── evaluate.py # Evaluation: confusion matrix, classification report, Grad-CAM
│ └── gradcam_utils.py # Grad-CAM implementation
├── split_dataset.py # Splits a raw downloaded dataset into train/val folders
├── requirements.txt
└── README.md

## How to Run Locally

```bash
git clone https://github.com/laxmipatil-lax/plant-disease-classifier.git
cd plant-disease-classifier
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Download the PlantVillage dataset from Kaggle, then split it:
python split_dataset.py --source path/to/downloaded/dataset --dest data --val-ratio 0.2

# 2. Train
python src/train.py --data-dir data --epochs 10

# 3. Evaluate (confusion matrix, per-class metrics, Grad-CAM)
python src/evaluate.py --data-dir data --checkpoint outputs/best_model.pt

# 4. Run the demo app
streamlit run app.py
```

## What's Next

- [ ] Finish full training run and add accuracy/F1 results here
- [ ] Add confusion matrix and Grad-CAM example images
- [ ] Deploy the Streamlit demo publicly (Streamlit Community Cloud / Hugging Face Spaces)
- [ ] Try a larger backbone (EfficientNet) and compare performance

## What I Learned

Building this project meant working through the full ML pipeline end-to-end — from dataset preparation and transfer learning to model interpretability with Grad-CAM and deployment planning. It reinforced how much of real ML work is data handling and infrastructure, not just model code.
