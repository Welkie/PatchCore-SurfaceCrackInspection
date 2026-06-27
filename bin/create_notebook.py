import os
import json

def generate_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Cold-Start Anomaly Detection for Concrete Surface Crack Inspection using PatchCore\n",
                    "\n",
                    "**Author:** Lam Huynh Khang — SE200666 — FPT University HCMC  \n",
                    "**Supervisor:** Nguyen Xuan Huy, Lecturer, FPT University  \n",
                    "\n",
                    "This notebook implements the complete baseline and experimental pipeline for the project proposal **Cold-Start Anomaly Detection for Concrete Surface Crack Inspection using PatchCore with Pixel-Level Heatmap Localization**. It covers the evaluation of unsupervised PatchCore against supervised CNN classifiers.\n",
                    "\n",
                    "### Objectives & Research Questions:\n",
                    "- **RQ1 (Detection reliability):** Detect cracks reliably with 0 defect images during training using PatchCore.\n",
                    "- **RQ2 (Pixel localization):** Localize crack regions via anomaly heatmaps compared against pseudo-ground-truth masks.\n",
                    "- **RQ3 (Coreset ablation):** Test how coreset subsampling ratio (1%, 5%, 10%, 25%, 50%, 100%) impacts AUROC and speed.\n",
                    "- **RQ4 (Data efficiency):** Compare PatchCore (0 training defects) against supervised classifiers (5, 10, 50, 100, 200 training defects)."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Setup & Dependencies\n",
                    "Install the required libraries, including `timm` for backbones and `faiss-gpu` for optimized nearest-neighbor searching."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install PyTorch, timm, faiss-gpu, and scikit-learn\n",
                    "!pip install timm faiss-gpu scikit-image scikit-learn matplotlib tqdm pillow click pandas"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Clone Repository or Verify Directory Structure\n",
                    "We need to make sure we are inside the repository workspace containing `src/patchcore`, `bin/run_experiments.py`, and the baseline scripts."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# If running on Kaggle, clone the repository containing the PatchCore code:\n",
                    "# !git clone https://github.com/<YOUR_GITHUB_USERNAME>/patchcore-inspection.git\n",
                    "# %cd patchcore-inspection\n",
                    "\n",
                    "import os\n",
                    "import sys\n",
                    "\n",
                    "# Add src/ and bin/ to python path\n",
                    "sys.path.append(os.path.abspath('src'))\n",
                    "sys.path.append(os.path.abspath('bin'))\n",
                    "\n",
                    "print(\"Current directory:\", os.getcwd())\n",
                    "print(\"Files in directory:\", os.listdir('.'))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Verify Kaggle Dataset Path\n",
                    "The Kaggle Surface Crack Detection dataset must be added to the notebook. We verify that the dataset path is accessible and check for the `Negative` and `Positive` folders."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "data_path = \"/kaggle/input/surface-crack-detection\"\n",
                    "if not os.path.exists(data_path):\n",
                    "    # Fallback to check nested folder structure if Kaggle extracts differently\n",
                    "    nested_path = \"/kaggle/input/surface-crack-detection/surface-crack-detection\"\n",
                    "    if os.path.exists(nested_path):\n",
                    "        data_path = nested_path\n",
                    "        \n",
                    "if not os.path.exists(data_path):\n",
                    "    raise ValueError(\"Dataset path not found. Please add arunrk7/surface-crack-detection to your Kaggle Notebook input.\")\n",
                    "else:\n",
                    "    print(f\"Found dataset at: {data_path}\")\n",
                    "    print(\"Subdirectories:\", os.listdir(data_path))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Execute End-to-End Experiments (Ablation & Baselines)\n",
                    "We run `bin/run_experiments.py` which automates:\n",
                    "1. Fitting and evaluating PatchCore on WideResNet-50 and ResNet-18 for all coreset ratios (1%, 5%, 10%, 25%, 50%, 100%).\n",
                    "2. Training and evaluating supervised classifiers (ResNet-50, EfficientNet-B0) on varying positive training samples (5, 10, 50, 100, 200).\n",
                    "3. Pre-generating comparative plots and saving all results under the `results/` folder."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run end-to-end experiment pipeline\n",
                    "!python bin/run_experiments.py --data_path {data_path} --results_path results --gpu 0"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. View Quantitative Results\n",
                    "Load the results tables into Pandas DataFrames to analyze the performance of PatchCore and the supervised baselines."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "from IPython.display import display, HTML\n",
                    "\n",
                    "print(\"=== TABLE 1: PATCHCORE CORESET ABLATION RESULTS (RQ3) ===\")\n",
                    "df_pc = pd.read_csv(\"results/patchcore_ablation_results.csv\")\n",
                    "display(df_pc)\n",
                    "\n",
                    "print(\"\\n=== TABLE 2: SUPERVISED BASELINE DATA EFFICIENCY RESULTS (RQ4) ===\")\n",
                    "df_sup = pd.read_csv(\"results/supervised/results.csv\")\n",
                    "display(df_sup)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Scientific Visualization Plots\n",
                    "Render the generated evaluation plots directly in the notebook to visualy answer RQ3 and RQ4."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from PIL import Image\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    "plots = [\n",
                    "    (\"results/plot_auroc_vs_coreset.png\", \"Coreset Subsampling Impact on Image AUROC\"),\n",
                    "    (\"results/plot_latency_vs_coreset.png\", \"Inference Speed vs. Coreset Subsampling Ratio\"),\n",
                    "    (\"results/plot_data_efficiency.png\", \"Data Efficiency Curve: Unsupervised PatchCore vs. Supervised CNNs\")\n",
                    "]\n",
                    "\n",
                    "for p, title in plots:\n",
                    "    if os.path.exists(p):\n",
                    "        print(f\"\\n=== {title} ===\")\n",
                    "        img = Image.open(p)\n",
                    "        plt.figure(figsize=(10, 6))\n",
                    "        plt.imshow(img)\n",
                    "        plt.axis('off')\n",
                    "        plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Qualitative visual heatmaps (RQ2)\n",
                    "Display the qualitative visual localization. For each tested backbone, this shows the Input Image, the generated Pseudo-Ground-Truth Mask, and the resulting PatchCore Anomaly Heatmap."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "for backbone in [\"resnet18\", \"wideresnet50\"]:\n",
                    "    p = f\"results/visual_localization_{backbone}.png\"\n",
                    "    if os.path.exists(p):\n",
                    "        print(f\"\\n=== Qualitative Crack Heatmap Overlay ({backbone}) ===\")\n",
                    "        img = Image.open(p)\n",
                    "        plt.figure(figsize=(12, 10))\n",
                    "        plt.imshow(img)\n",
                    "        plt.axis('off')\n",
                    "        plt.show()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    # Save to the root of the workspace
    target_notebook = "d:/My document/CPV301/patchcore-inspection/PatchCore_Concrete_Crack_Kaggle.ipynb"
    with open(target_notebook, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"Generated notebook at: {target_notebook}")

if __name__ == "__main__":
    generate_notebook()
