# Machine Learning for Higgs Boson Classification at the LHC

This project develops and evaluates a machine learning pipeline for Higgs boson signal classification in the (VH \rightarrow Vb\bar{b}) channel using simulated LHC collision data. The main goal is to compare a traditional cut-based analysis with a feed-forward neural network and assess both with a physics-driven metric: the expected discovery sensitivity (Asimov significance).

The final optimised neural network reaches **(Z_{\mathrm{NN}} = 2.305 \pm 0.020)**, compared with **(Z_{\mathrm{cut}} = 1.897)** for the optimised cut-based baseline, corresponding to a **21.5% improvement** in sensitivity.

## Project overview

The workflow includes:

* **Data preparation**: loading train/validation/test datasets, cleaning unnamed columns, checking data consistency
* **Feature engineering and physics variable study**: using a compact set of physics-motivated kinematic observables
* **Cut-based baseline**: sequential threshold scan to construct a reference analysis
* **Neural network modelling**: training a fully connected classifier on the same feature set
* **Physics-driven evaluation**: using adaptive binning and Asimov significance instead of standard accuracy alone
* **Hyperparameter optimisation**: sequential scans plus a reduced-space comparison of grid, random, and Bayesian search
* **Extended studies**: training-statistics scaling, feature ablation, repeated-training uncertainty, high-score region analysis, and robustness under input distortion

## Input features

The model uses five kinematic variables:

* `mBB` — invariant mass of the two (b)-tagged jets
* `dRBB` — angular separation between the two (b)-jets
* `MET` — missing transverse energy
* `Mtop` — reconstructed top-quark mass
* `pTV` — transverse momentum of the vector boson candidate

These variables are used in both the cut-based and neural-network analyses.

## Main results

* **No-cut baseline sensitivity**: (Z_0 = 1.496)
* **Optimised cut-based sensitivity**: (Z_{\mathrm{cut}} = 1.897)
* **Baseline neural network sensitivity**: (Z_{\mathrm{NN,base}} = 2.196 \pm 0.020)
* **Final optimised neural network sensitivity**: (Z_{\mathrm{NN}} = 2.305 \pm 0.020)
* **Improvement over cut-based baseline**: **21.5%**


## Dependencies

The notebook uses the following main Python libraries:

* `numpy`
* `pandas`
* `matplotlib`
* `scikit-learn`
* `tensorflow / keras`
* `scipy`

A minimal `requirements.txt` can be:

```txt
numpy
pandas
matplotlib
scikit-learn
tensorflow
scipy
jupyter
```

## How to run

### 1. Clone the repository

```bash
git clone https://github.com/<YutongWang1031>/<higgs-signal-neural-network-classification>.git
cd <higgs-signal-neural-network-classification>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the data

Place the CSV files in your `data/` folder and update the dataset path in the notebook if needed.

In the original notebook, the data path is set through:

```python
base_path = "/content/drive/MyDrive/ML_project/data-v2/"
```

If you are running locally, replace it with something like:

```python
base_path = "./data/"
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

Then open:

```text
Project_Python_Code_LHC_Higgs.ipynb
```

## Notebook workflow

The notebook is organised into the following main sections:

### 1. Dataset and variables

* loads the datasets
* removes unnamed columns
* checks for missing values and consistency
* visualises the main physics variables

### 2. Cut-based analysis

* computes the no-cut baseline
* performs sequential cut scans on:

  * `Mtop`
  * `dRBB`
  * `pTV`
  * `MET`
* builds the final cut-based selection

### 3. Neural network analysis

* prepares scaled train/validation/test inputs
* defines the feed-forward neural network
* evaluates the baseline NN
* performs hyperparameter scans:

  * hidden units
  * number of layers
  * optimiser
  * activation
  * loss
  * scaler
* performs training-parameter scans:

  * epochs
  * batch size
  * learning rate
* evaluates the final optimised model on the test set

### 4. Performance studies

* compares cut-based and NN sensitivity
* studies training-data scaling
* performs leave-one-out feature importance analysis
* estimates training uncertainty across repeated runs
* examines the high-NN-score region
* studies robustness under `mBB` distortion
* compares grid, random, and Bayesian hyperparameter search on a reduced search space

## Evaluation metric

This project does **not** rely on classification accuracy alone. Model performance is evaluated using a **binned Asimov significance**, which is more appropriate for the physics objective of signal discovery.

The neural-network output is treated as a continuous discriminant, adaptively binned, and then converted into a signal sensitivity estimate with uncertainty.

## Hyperparameter optimisation strategy

The optimisation is performed in two stages:

1. **Sequential scans**
   Used as an initial step to identify a promising region of parameter space under limited computing resources.

2. **Reduced-space joint comparison**
   A smaller candidate space is then explored using:

   * **grid search**
   * **random search**
   * **Bayesian search**

Within this reduced space, grid search gave the strongest result, while the final sequentially optimised model remained competitive at test level.

## Robustness and diagnostics

To go beyond a single best-model result, the project includes several diagnostic studies:

* **training-statistics scaling**
  to test whether more data continues to improve performance

* **feature ablation**
  to identify which variables drive the classifier

* **repeated retraining**
  to estimate stochastic variation from random initialisation

* **distribution distortion tests**
  to evaluate sensitivity to mismodelling in the most important input variable (`mBB`)

These studies help distinguish genuine model improvement from unstable or overly specific behaviour.

## Notes on reproducibility

Some parts of the project, especially the hyperparameter search comparison, are computationally expensive. In the original notebook, a note is included that Colab session limits may interrupt long runs. If needed, save intermediate outputs or summary tables separately.

The notebook also references a support file:

```text
ucl_masterclass.py
```

If you plan to publish this repository, make sure that helper file is included or clearly document which utilities from it are required.

## Report

The full project report is available in:

```text
report/Mini-project_Report_LHC_Higgs.pdf
```

It includes the full methodology, figures, performance studies, and discussion of limitations.

## Skills demonstrated

This project demonstrates experience in:

* **Python data analysis**
* **machine learning model development**
* **feature engineering**
* **hyperparameter optimisation**
* **custom metric design**
* **robustness testing**
* **scientific data interpretation**
* **working with structured experimental / simulation data**


## Author

**Yutong Wang**
UCL Physics and Astronomy
PHAS0056: Practical Machine Learning for Physicists Mini-project

