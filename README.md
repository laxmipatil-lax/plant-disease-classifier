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
