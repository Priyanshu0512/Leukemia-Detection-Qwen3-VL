# Leukemia Detection using Qwen3-VL and ResNet50 Ensemble

A comprehensive deep learning project for automated leukemia detection from blood smear images using an ensemble approach combining ResNet50 (CNN) and Qwen3-VL (Vision-Language Model). This project achieves state-of-the-art performance with 98.78% accuracy on test data.

[![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.57+-FFD700?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=google-colab&logoColor=white)](https://colab.research.google.com/)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Technical Details](#technical-details)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a hybrid ensemble model for leukemia classification from blood smear microscopy images. It combines:

1. **ResNet50**: A deep convolutional neural network fine-tuned for image classification
2. **Qwen3-VL-2B-Instruct**: A vision-language model that uses natural language prompts for classification

The ensemble approach leverages the strengths of both models to achieve superior performance compared to individual models.

### Classification Classes

The model classifies blood smear images into four categories:
- **Benign**: Normal blood cells
- **Early**: Early-stage leukemia
- **Pre**: Pre-leukemia stage
- **Pro**: Pro-leukemia stage

## ✨ Features

- **Dual-Model Ensemble**: Combines CNN and Vision-Language Model predictions
- **Stratified Data Splitting**: 70/15/15 train/validation/test split maintaining class balance
- **Transfer Learning**: Pre-trained ResNet50 fine-tuned on leukemia dataset
- **Sampling-Based Probability Estimation**: Qwen3-VL uses multiple samples for robust predictions
- **Comprehensive Evaluation**: Detailed metrics including precision, recall, F1-score, and confusion matrices
- **Google Colab Optimized**: Designed for GPU-accelerated training in Colab environment
- **Kaggle Dataset Integration**: Easy dataset download and preprocessing

## 🏗️ Architecture

### Model Components

1. **ResNet50 CNN**
   - Pre-trained on ImageNet
   - Fine-tuned with transfer learning
   - Input size: 224×224 pixels
   - Data augmentation: Random horizontal flip, rotation (±10°)
   - Optimizer: Adam with learning rate 1e-4
   - Training epochs: 8

2. **Qwen3-VL-2B-Instruct**
   - Vision-Language Model from Qwen
   - Uses expert hematologist prompts
   - Sampling-based probability estimation (6 samples per image)
   - Temperature: 0.9 for generation diversity

3. **Ensemble Fusion**
   - Weighted combination: `α × CNN_probs + (1-α) × Qwen_probs`
   - Optimal α found via grid search (typically 0.5)
   - Final prediction: argmax of fused probabilities

## 📊 Dataset

The project uses a Kaggle dataset containing blood smear images organized by leukemia stages. The dataset structure:

```
kaggle_data/
├── Segmented/          # Pre-processed segmented images (preferred)
│   ├── Benign/
│   ├── Early/
│   ├── Pre/
│   └── Pro/
└── Original/           # Original images (fallback)
    ├── Benign/
    ├── Early/
    ├── Pre/
    └── Pro/
```

### Data Splitting

- **Training**: 70% of data
- **Validation**: 15% of data
- **Test**: 15% of data

The split is stratified to maintain class distribution across all splits.

## 🚀 Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended)
- Google Colab account (for cloud execution)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Leukemia-Detection-Qwen3-VL-master
   ```

2. **Install dependencies** (run in Google Colab or local environment)
   ```python
   !pip install -q kaggle
   !pip install -q "transformers>=4.57.0" accelerate
   !pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   !pip install -q scikit-learn pandas matplotlib seaborn pillow tqdm
   ```

3. **Set up Kaggle API** (for dataset download)
   - Upload `kaggle.json` to Colab
   - Or configure Kaggle credentials manually

## 💻 Usage

### Running in Google Colab

1. **Open the notebook**
   - Upload `LeukemiaDetection.ipynb` to Google Colab
   - Ensure GPU runtime is enabled (Runtime → Change runtime type → GPU)

2. **Execute cells sequentially**

   **Cell 1.1**: Check GPU availability
   ```python
   import torch
   print("Torch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
   ```

   **Cell 1.2**: Install required packages (see Installation section)

   **Cell 1.3**: Mount Google Drive (optional, for saving models)
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

   **Cell 2**: Download Kaggle dataset
   - Configure Kaggle API credentials
   - Download and extract dataset

   **Cell 3**: Data preprocessing and stratified split
   - Automatically detects Segmented or Original images
   - Creates train/val/test splits (70/15/15)

   **Cell 4**: Train ResNet50 model
   - Fine-tunes ResNet50 on training data
   - Saves best model checkpoint
   - Generates test predictions and probabilities

   **Cell 5**: Load Qwen3-VL model
   - Downloads and initializes Qwen3-VL-2B-Instruct
   - Sets up classification functions

   **Cell 6**: Compute Qwen3-VL probabilities
   - Processes validation and test sets
   - Saves probability distributions

   **Cell 7**: Ensemble fusion and evaluation
   - Combines CNN and Qwen3-VL predictions
   - Finds optimal ensemble weight (α)
   - Generates final classification report

### Key Parameters

- **Batch size**: 32
- **Learning rate**: 1e-4
- **Epochs**: 8
- **Qwen samples**: 6 per image
- **Temperature**: 0.9
- **Input size**: 224×224

## 📁 Project Structure

```
Leukemia-Detection-Qwen3-VL-master/
│
├── LeukemiaDetection.ipynb      # Main Jupyter notebook
├── README.md                     # This file
│
├── dataset/                      # Processed dataset (created during execution)
│   ├── train/
│   │   ├── Benign/
│   │   ├── Early/
│   │   ├── Pre/
│   │   └── Pro/
│   ├── val/
│   └── test/
│
├── cnn_test_results.pkl          # ResNet50 test predictions
├── best_resnet50.pth            # Trained ResNet50 model checkpoint
│
├── qwen_results/                 # Qwen3-VL outputs
│   ├── qwen_val_results.pkl
│   └── qwen_test_results.pkl
│
└── ensemble_results_test.csv     # Final ensemble predictions
```

## 📈 Results

### Performance Metrics

The ensemble model achieves the following performance on the test set:

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **98.78%** |
| **Best Ensemble α** | 0.5 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Benign | 0.96 | 0.96 | 0.96 | 77 |
| Early | 0.99 | 0.99 | 0.99 | 149 |
| Pre | 0.99 | 1.00 | 1.00 | 145 |
| Pro | 1.00 | 0.98 | 0.99 | 122 |

### Confusion Matrix

```
              Predicted
Actual    Benign  Early  Pre  Pro
Benign      74      2     1    0
Early        1    148     0    0
Pre          0      0   145    0
Pro          2      0     0  120
```

### Ensemble Weight Analysis

| α (CNN Weight) | Accuracy |
|----------------|----------|
| 0.0 (Qwen only) | 20.69% |
| 0.25 | 48.28% |
| **0.5** | **98.78%** |
| 0.75 | 98.58% |
| 1.0 (CNN only) | 98.58% |

## 🔧 Technical Details

### ResNet50 Training

- **Architecture**: ResNet50 with ImageNet pre-trained weights
- **Final Layer**: Custom fully-connected layer (2048 → 4 classes)
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam optimizer
- **Learning Rate**: 1e-4
- **Data Augmentation**:
  - Random horizontal flip (50% probability)
  - Random rotation (±10 degrees)
  - ImageNet normalization

### Qwen3-VL Classification

- **Model**: Qwen3-VL-2B-Instruct
- **Prompt Template**: Expert hematologist classification prompt
- **Sampling Strategy**: 6 independent samples per image
- **Probability Estimation**: Frequency-based from sampled outputs
- **Generation Parameters**:
  - Temperature: 0.9
  - Top-p: 0.95
  - Max new tokens: 8

### Ensemble Method

The ensemble combines predictions using weighted averaging:

```
P_ensemble = α × P_CNN + (1 - α) × P_Qwen
```

Where:
- `P_CNN`: Probability distribution from ResNet50
- `P_Qwen`: Probability distribution from Qwen3-VL
- `α`: Ensemble weight (optimized via grid search)

## 📦 Requirements

### Python Packages

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.57.0
accelerate
scikit-learn
pandas
matplotlib
seaborn
pillow
tqdm
kaggle
```

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended)
  - Minimum: 4GB VRAM
  - Recommended: 8GB+ VRAM (T4 or better)
- **RAM**: 8GB+ recommended
- **Storage**: ~5GB for dataset and models

### Software Requirements

- Python 3.7+
- CUDA 12.1+ (for GPU acceleration)
- Jupyter Notebook or Google Colab

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Areas for Improvement

- Support for additional vision-language models
- Real-time inference API
- Web-based interface for predictions
- Model explainability and visualization
- Additional data augmentation techniques

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Qwen Team** for the Qwen3-VL vision-language model
- **PyTorch Team** for the deep learning framework
- **Kaggle** for hosting the leukemia detection dataset
- **Google Colab** for providing free GPU resources

---

**Note**: This project is designed for research and educational purposes. For medical diagnosis, always consult qualified healthcare professionals.
