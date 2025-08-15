# Bidirectional Temporal-Aware Modeling with Multi-Scale Mixture-of-Experts for Multivariate Time Series Forecasting

This code is the official PyTorch implementation of CIKM'25 paper: Bidirectional Temporal-Aware Modeling with Multi-Scale Mixture-of-Experts for Multivariate Time Series Forecasting ( $\text{BIM}^3$ ).

## Quickstart

### 1. Environment

$\text{BIM}^3$ is developed with Python 3.10 and relies on Pytorch 2.4.1. To set up the environment, make sure miniconda has been correctly installed and configed.

```bash
# Create a new conda environment.
conda create -n bim3 python=3.10 -y
conda activate bim3

# Install required packages using pip.
pip install -r requirements.txt
```

### 2. Dataset

$\text{BIM}^3$ adopts [TFB](https://github.com/decisionintelligence/TFB) framework as the code basis, and datasets can be obtained from TFB's public archive [Google Drive](https://drive.google.com/file/d/1vgpOmAygokoUt235piWKUjfwao6KwLv7/view?usp=drive_link).
Place the downloaded .zip file under the folder `./dataset`.
The structure of folders should be:

```plain
.
├── config
├── dataset
│   └── forecasting
|       ├── ...
|       ├── Electricity.csv
|       ├── ETTh1.csv
|       ...
...
```

### 3. Run Scripts

We provide all experiment scripts for $\text{BIM}^3$ for 10 datasets (Electricity, ETTh1, ETTh2, ETTm1, ETTm2, Exchange, ILI, Solar, Traffic and Weather). These scripts are placed under folder `./scripts/multivariate_forecast`. For instance, you can reproduce the ETTh1.csv dataset's results by running:

```bash
sh ./scripts/multivariate_forecast/ETTh1_script/BIM3.sh
```

The experiments results would be under the folder `results/ETTh1`, which are stored in .csv format with detailed training configuration. All same to other datasets' scripts.

### 4. Code Implementation

The implementation of $\text{BIM}^3$ can be found under folder `ts_benchmark/baselines/bim3`.

### 5. Citation

If you find this repo is helpful, please cite our paper.

```
Blablabla
```
