<div align="center">

# 🩺 SkinScan Doctor
### Skin Disease Classification with Swin Transformer

**A deep-learning computer vision project for classifying dermatological images into 23 skin-disease categories.**

[![Python](https://img.shields.io/badge/Python-Deep%20Learning-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Model%20Training-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/Model-timm-6C5CE7)](https://huggingface.co/docs/timm/index)
[![Task](https://img.shields.io/badge/Task-23--Class%20Classification-0A9396)](#-project-overview)
[![Dataset](https://img.shields.io/badge/Dataset-DermNet-457B9D)](#-dataset)

</div>

---

## 🧠 Project overview

**SkinScan Doctor** explores the use of a pretrained **Swin Transformer** for multi-class classification of dermatological images. The project uses PyTorch and `timm` to train a Swin-Tiny model on a 23-class DermNet dataset, with preprocessing, data augmentation, validation, and test-set evaluation.

The goal is to build a computer-vision pipeline that can recognise visual patterns associated with different skin-condition categories and serve as a foundation for an accessible inference experience.

> **Medical-use disclaimer:** This is an educational machine-learning project, not a medical device. Model predictions are not a diagnosis and must not replace assessment by a qualified healthcare professional. Performance can vary across image quality, populations, devices, and clinical settings.

## ✨ Highlights

- 🧠 **Swin Transformer:** Transfer learning with the pretrained `swin_tiny_patch4_window7_224` architecture.
- 🏷️ **23 output classes:** Multi-class prediction across the dataset's skin-condition categories.
- 🧪 **Training pipeline:** Separate train, validation, and test splits.
- 🖼️ **Image preprocessing:** Resize images to 224 × 224 and apply ImageNet normalization.
- 🔄 **Training augmentation:** Random horizontal flips, rotations, and brightness/contrast/saturation variation.
- ⚙️ **Optimisation:** AdamW, cross-entropy loss, cosine learning-rate scheduling, and mixed-precision training.
- 📊 **Evaluation:** Test accuracy, classification report, and confusion-matrix visualisation are included in the notebook.
- 🚀 **Deployment direction:** The project description references Hugging Face Spaces for inference. See the deployment section for the repository artefacts currently available.

## 🔬 Model and workflow

```text
DermNet image dataset
          │
          ▼
  Train / validation / test
          │
          ▼
  Resize to 224 × 224
  Normalize with ImageNet statistics
          │
          ├── Training only:
          │   flips, rotations, color jitter
          │
          ▼
 Pretrained Swin-Tiny Transformer
          │
          ▼
     23-class classifier
          │
          ▼
 Validation & model selection
          │
          ▼
 Test evaluation
          │
          ▼
 Accuracy · Classification report · Confusion matrix
```

### Why Swin Transformer?

Swin Transformer is a vision-transformer architecture that computes attention within local windows and shifts windows between layers. This hierarchical design helps it model visual patterns at multiple spatial scales while controlling attention cost compared with global attention over every image patch.

In this project, `timm` creates the pretrained Swin-Tiny model and replaces its classification head with one sized for the dataset's **23 classes**.

## 🗂️ Dataset

The notebook expects the images to be arranged in class-specific directories, compatible with `torchvision.datasets.ImageFolder`:

```text
dataset/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── val/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

The notebook output shows the 23 DermNet category names and split counts for the original dataset arrangement:

| Split | Images |
|---|---:|
| Train | 13,233 |
| Validation | 4,012 |
| Test | 4,017 |
| **Total** | **21,262** |

The notebook also contains experiments referencing an augmented dataset directory, with different split counts. Treat dataset totals as belonging to the specific experiment/split configuration rather than combining those counts.

### Preprocessing

- Resize images to **224 × 224**.
- Convert images to tensors.
- Normalize using ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
- Apply random horizontal flipping, rotation up to 15 degrees, and colour jitter to training images.
- Apply deterministic resizing and normalization to validation and test images.

**Dataset access:** The data directory is expected locally and is not included in the repository listing. Obtain DermNet data through its appropriate source and ensure your use complies with its licence and terms.

## 📈 Results and evaluation

### Validation performance — Swin-Tiny

The training screenshot for the 25-epoch run shows validation accuracy holding at approximately **83%** in the final three epochs.

| Epoch | Training accuracy | Training loss | Validation accuracy | Validation loss |
|---|---:|---:|---:|---:|
| 23 / 25 | 97.44% | 0.0621 | **83.10%** | 0.7996 |
| 24 / 25 | 97.38% | 0.0646 | 83.00% | 0.8000 |
| 25 / 25 | 97.47% | 0.0614 | 83.05% | 0.8008 |

**Best validation accuracy shown: 83.10% at epoch 23.** This is a validation metric, not a test-set result. The final three epochs show a persistent gap between training accuracy (~97.4%) and validation accuracy (~83%), which suggests overfitting and supports selecting checkpoints based on validation performance rather than simply using the final epoch.

### Test performance — a separately recorded notebook run

The committed notebook also contains a test evaluation output reporting **67.39% test accuracy** with a test loss of **1.8979**, along with a per-class classification report and confusion-matrix code.

| Metric | Recorded result | Context |
|---|---:|---|
| Best validation accuracy in the supplied training screenshot | **83.10%** | Epoch 23 of a 25-epoch run |
| Test accuracy in the committed notebook output | **67.39%** | Separately recorded evaluation run |

These values refer to different recorded results and should not be presented as if they were measured on the same run or checkpoint. The notebook contains experiments using different dataset directory configurations, so comparisons require matching the checkpoint, data split, preprocessing, and class mapping. Until the exact run/checkpoint relationship is verified, report the **83.10% as validation accuracy** and the **67.39% as the test result shown in the committed notebook output**.

Evaluation includes:
- Overall test accuracy and loss.
- Per-class precision, recall, and F1-score.
- A confusion matrix to inspect common class confusions.

For medical image classification, aggregate accuracy alone is not sufficient. Per-class recall, precision, class imbalance, independent test data, and external validation matter—particularly where different classes may have different clinical consequences.

## 🧰 Technology stack

| Area | Tools |
|---|---|
| Language | Python |
| Deep learning | PyTorch |
| Vision models | `timm` / Swin Transformer |
| Data loading | `torchvision.datasets.ImageFolder` |
| Augmentation | `torchvision.transforms` |
| Training | AdamW, Cross-Entropy, CosineAnnealingLR |
| Efficiency | CUDA when available, automatic mixed precision |
| Evaluation | scikit-learn classification report/confusion matrix, Matplotlib, Seaborn |
| Notebook | Jupyter Notebook |

## ▶️ Run the notebook

### 1. Clone the repository

```bash
git clone https://github.com/Jayantsinghkhanna/23-Skin-diseases-detection-using-Swin-transformer-.git
cd 23-Skin-diseases-detection-using-Swin-transformer-
```

### 2. Create an environment

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### 3. Install the notebook dependencies

```bash
python -m pip install --upgrade pip
pip install torch torchvision timm tqdm matplotlib pandas scikit-learn seaborn jupyter
```

Install a PyTorch build that matches your CUDA environment if using an NVIDIA GPU. The notebook automatically selects CUDA when available, otherwise it falls back to CPU.

### 4. Prepare the dataset

Create the `dataset/train`, `dataset/val`, and `dataset/test` folders and place images in subfolders named by class. Update `DATA_ROOT` in the notebook if your dataset path differs.

### 5. Launch Jupyter

```bash
jupyter notebook 23_Skin_Diseases_detection.ipynb
```

Run the cells in order. Training can be computationally intensive; a CUDA-capable GPU is recommended.

## 📁 Repository structure

```text
.
├── 23_Skin_Diseases_detection.ipynb   # Training, validation, and evaluation workflow
└── README.md                          # Project guide and methodology
```

The notebook expects the dataset locally. Model checkpoints and a standalone inference application are not present in the repository file listing at the time this README was updated.

## 🚀 Deployment

The project description mentions Hugging Face Spaces as the intended inference deployment. A deployable Space typically needs an inference entry point, dependency specification, label mapping, model weights, and preprocessing that exactly matches training.

The current repository listing contains the notebook and README only, so this repository alone does not expose those deployment artefacts. Add a link to the live Space and document its inference interface here once the deployment is available and verified.

## 🛣️ Potential improvements

- Save and version the best checkpoint based on validation performance.
- Re-run test evaluation using the checkpoint selected by validation performance and document the matching dataset split/configuration.
- Add class-wise metrics and normalized confusion matrix images to the README.
- Track experiments and model configurations for reproducibility.
- Evaluate on an independent dataset and check for dataset leakage and distribution shift.
- Add a validated inference script or Hugging Face Spaces app with a clear model card.
- Consider explainability tools such as Grad-CAM or attention visualisations, while validating that visual explanations are meaningful.

---

<div align="center">

**SkinScan Doctor** · Exploring transformer-based computer vision for dermatological image classification.

*Research and education only — not for clinical diagnosis or treatment decisions.*

</div>
