# Neural Collapse & OOD Detection on CIFAR-100

This project explores the emergence of **Neural Collapse (NC)** in deep classifiers and its impact on **Out-of-Distribution (OOD)** detection. You can find more details in the provided report: `./report.pdf`

## Quick Start

### 1. Requirements
Ensure you have the following installed:
* Python 3.10+
* requirements.txt (`pip install -r requirements.txt`)

### 2. Configuration
Before running the notebooks, verify the paths in `config.py`. This file centralizes the environment settings so you don't have to modify the source code:

```python
# config.py
DATA_DIR = "./data". # Path to CIFAR-100 and SVHN (auto download if doesn't exist)
MODELS_DIR = "./models_weights"  # Where to load/save .pth files
BATCH_SIZE = 128
```

## Project Structure

* **`notebooks/nc_analysis.ipynb`**: Evaluation of Neural Collapse properties (NC1 to NC5) and PCA visualizations of the latent space.
* **`notebooks/compare_ood.ipynb`**: Benchmarking OOD scoring methods (MSP, Energy, ViM, and NECO) within the induced collapse regime.
* **`notebooks/train_cifar.ipynb`**: Training ResNet on CIFAR-100 dataset.

* **`src/`**: Core logic including model definitions, geometric loss functions, and OOD metrics.
* **`figures/`**: Generated plots for latent space organization and metric distributions.
* **`data/`**: Where the are stored (auto downloaded if needed)
* **`models_weights/`**: Last version of the models weights (provided in the github repository)
