🌱 Plant Pathology 2021 FGVC8 – Multi-Label Image Classification

This project implements multi-label classification for plant disease images from the Plant Pathology 2021 FGVC8 Challenge
.

Multiple CNN architectures are used for feature extraction and classification, including:

DenseNet121

MobileNetV2

EfficientNetV2B0

Predictions are thresholded using precision-recall optimization for each class, and a Kaggle submission file is generated.

🛠 Features

Load and preprocess images from directories.

Handle multi-label outputs using MultiLabelBinarizer.

Train pretrained CNNs (DenseNet121, MobileNetV2, EfficientNetV2B0) with fine-tuning.

Optimize per-class thresholds using precision-recall curves.

Generate submission-ready CSV files for Kaggle evaluation.

Evaluate models using classification report and macro F1 score.

⚡ Installation

Install required Python packages:

pip install tensorflow scikit-learn pandas numpy opencv-python pillow tqdm


Optional: Use GPU acceleration for faster training.

📝 Usage

Organize your dataset according to Kaggle challenge:

train_images/
test_images/
train.csv
sample_submission.csv


Set paths in the script:

train_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"
test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"
train_csv = "/kaggle/input/plant-pathology-2021-fgvc8/train.csv"


Run the script:

python plant_pathology_classification.py


A submission file submission.csv will be generated with predicted labels.

🔧 How It Works

Data Loading & Preprocessing:

Resize images to (224, 224).

Normalize pixel values to [0,1].

Label Encoding:

Multi-label binarization using MultiLabelBinarizer.

Modeling:

Fine-tune pretrained CNNs: DenseNet121, MobileNetV2, EfficientNetV2B0.

Add fully-connected layers for multi-label prediction.

Threshold Optimization:

Calculate optimal per-class thresholds using precision-recall curves.

Prediction & Submission:

Convert probabilities to binary predictions using optimized thresholds.

Transform binary vectors back to space-separated class strings.

Save results in submission.csv.

📂 Project Structure
plant-pathology-classification/
│
├─ plant_pathology_classification.py   # Main script
├─ README.md                           # Documentation

🧠 Example Output
DenseNet121 Macro F1 Score: 0.86
MobileNetV2 Macro F1 Score: 0.85
EfficientNetV2B0 Macro F1 Score: 0.87

🔗 Dependencies

Python 3.x

TensorFlow / Keras

NumPy

Pandas

OpenCV

Pillow

Scikit-learn

tqdm
