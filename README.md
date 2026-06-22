# Machine Learning for Higgs Boson Classification

This project compares a traditional cut-based analysis with a neural-network classifier for Higgs boson signal identification in the VH → Vbb channel using simulated LHC data.

The final optimised neural network reaches **ZNN = 2.305 ± 0.020**, compared with **Zcut = 1.897** for the cut-based baseline, corresponding to a **21.5% improvement** in sensitivity.

## Files

This repository contains:

* `Project_Python_Code_LHC_Higgs.ipynb` — main analysis notebook
* `Mini-project_Report_LHC_Higgs.pdf` — full project report
* `VHbb_data_2jet.csv.zip` — full dataset
* `VHbb_data_2jet_train.csv.zip` — training dataset
* `VHbb_data_2jet_val.csv.zip` — validation dataset
* `VHbb_data_2jet_test.csv` — test dataset

## Methods

The analysis includes:

* cut-based baseline optimisation
* feed-forward neural network classification
* physics-driven evaluation using **Asimov significance**
* hyperparameter optimisation with **sequential**, **grid**, **stochastic**, and **Bayesian** search
* feature importance analysis
* training-statistics study
* robustness tests under input distortion

## Main result

* **Cut-based baseline:** Zcut = 1.897
* **Final neural network:** ZNN = 2.305 ± 0.020
* **Improvement:** **21.5%**

## How to run

This notebook was written for **Google Colab** and expects the project files to be available in Google Drive.

### Recommended setup

1. Download or clone this repository
2. Upload the dataset files to your Google Drive
3. Open `Project_Python_Code_LHC_Higgs.ipynb` in **Google Colab**
4. Mount Google Drive in Colab
5. Update the variable `base_path` so that it points to the folder containing the CSV files

In the current notebook, the dataset path is:

```python
base_path = "/content/drive/MyDrive/ML_project/data-v2/"
```

If your files are stored elsewhere, only this path needs to be changed before running the data-loading cells.

### Notes

* Some files are stored as `.zip`; unzip them before use if needed.
* The hyperparameter search comparison is computationally expensive, so some outputs may take a long time to reproduce in Colab.
* A saved output PDF is included in the original project archive for convenience when Colab sessions disconnect.

## Requirements

Main Python packages used:

* `numpy`
* `pandas`
* `matplotlib`
* `scikit-learn`
* `tensorflow`
* `scipy`
* `jupyter`

## Project summary

This project demonstrates experience in:

* **Python data analysis**
* **machine learning model development**
* **feature engineering**
* **hyperparameter optimisation**
* **custom metric design**
* **robustness testing**
* **scientific data interpretation**

## Author

**Yutong Wang**
UCL Physics and Astronomy
