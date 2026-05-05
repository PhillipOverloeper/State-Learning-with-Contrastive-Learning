# IEEE CAI Experiments: Discrete Representation Learning for Time-Series Data

This repository contains the code for the experiments conducted for our IEEE CAI paper.

The code supports experiments with:

- SOM-VAE
- CatVAE
- SimCLR

For SOM-VAE and CatVAE, both base and contrastive variants are supported.

---

## Repository Structure

```text
.
├── main.py
├── utils.py
├── SOMVAE.py
├── CATVAE.py
├── SimCLR.py
├── requirements.txt
├── data/
│   └── raw/
│       ├── HAR/
│       ├── MHEALTH/
│       └── PAMAP2/
├── results/
└── README.md
```

---

## Installation

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Data

The raw datasets are not included in this repository. Please download the datasets separately and place them in the `data/raw/` directory.

The currently supported datasets are:

```text
HAR
MHEALTH
PAMAP2
```

The dataset is selected via the `--dataset` argument.

Example:

```bash
python main.py --model catvae --mode train --variant base --dataset MHEALTH
```

---

## Expected Data Format

The preprocessing code expects each dataset to be loaded as one or more pandas DataFrames. Each DataFrame represents one recording, subject or sequence.

The exact column handling depends on the dataset.

---

### MHEALTH

For MHEALTH, each DataFrame is expected to have the following structure:

```text
feature_1, feature_2, ..., feature_n, state
```

The last column is interpreted as the state label.

The preprocessing performs the following steps:

1. Downsamples the data by taking every fifth row.
2. Uses the last column as the state label.
3. Removes rows where the state label is `0`.
4. Normalises the feature values using z-score normalisation.
5. Uses the DataFrame index as the time vector.

The expected input dimension is:

```text
23
```

---

### HAR

For HAR, each DataFrame is expected to have the following structure:

```text
feature_1, feature_2, ..., feature_n, unused_column, state
```

The last column is interpreted as the state label. The second-to-last column is ignored by the current preprocessing code.

The preprocessing performs the following steps:

1. Uses all columns up to `:-2` as features.
2. Uses the last column as the state label.
3. Removes rows where the state label is `0`.
4. Normalises the feature values using z-score normalisation.
5. Uses the DataFrame index as the time vector.

The expected input dimension is:

```text
561
```

---

### PAMAP2

For PAMAP2, each DataFrame is expected to have the following structure:

```text
time_or_id, state, feature_1, feature_2, ..., feature_n
```

The second column is interpreted as the state label.

The preprocessing performs the following steps:

1. Uses column index `1` as the state label.
2. Uses columns from index `2` onward as features.
3. Removes rows where the state label is `0`.
4. Normalises the feature values using z-score normalisation.
5. Uses the DataFrame index as the time vector.

The expected input dimension is:

```text
51
```

---

## Running Experiments

Experiments are launched through `main.py`.

The general command structure is:

```bash
python main.py \
  --model <model> \
  --mode <train_or_tune> \
  --variant <base_or_contrastive> \
  --dataset <dataset>
```

Available models:

```text
somvae
catvae
simclr
```

Available modes:

```text
train
tune
```

Available variants:

```text
base
contrastive
```

---

## CatVAE

### Train Base CatVAE

```bash
python main.py \
  --model catvae \
  --mode train \
  --variant base \
  --dataset MHEALTH \
  --seeds 42 43 44 \
  --enc-out-dim 16 \
  --dec-out-dim 8 \
  --cat-dim 20 \
  --beta 1.0 \
  --gumbel-temperature 0.9 \
  --learning-rate 0.001 \
  --num-epochs 100
```

### Train Contrastive CatVAE

```bash
python main.py \
  --model catvae \
  --mode train \
  --variant contrastive \
  --dataset MHEALTH \
  --seeds 42 43 44 \
  --enc-out-dim 16 \
  --dec-out-dim 8 \
  --cat-dim 20 \
  --beta 1.0 \
  --gumbel-temperature 0.9 \
  --noise 5e-3 \
  --temperature 0.1 \
  --learning-rate 0.001 \
  --num-epochs 100
```

### Tune Base CatVAE

```bash
python main.py \
  --model catvae \
  --mode tune \
  --variant base \
  --dataset HAR \
  --seeds 42 43 44
```

### Tune Contrastive CatVAE

```bash
python main.py \
  --model catvae \
  --mode tune \
  --variant contrastive \
  --dataset PAMAP2 \
  --seeds 42 43 44
```

---

## SOM-VAE

### Train Base SOM-VAE

```bash
python main.py \
  --model somvae \
  --mode train \
  --variant base \
  --dataset MHEALTH \
  --seeds 42 43 44 \
  --latent 16 \
  --hidden 16 \
  --som-dim 2 2 \
  --alpha 1.0 \
  --beta 1.0 \
  --gamma 0.8 \
  --tau 0.8 \
  --learning-rate 0.001 \
  --num-epochs 100
```

### Train Contrastive SOM-VAE

```bash
python main.py \
  --model somvae \
  --mode train \
  --variant contrastive \
  --dataset MHEALTH \
  --seeds 42 43 44 \
  --latent 16 \
  --hidden 16 \
  --som-dim 2 2 \
  --alpha 1.0 \
  --beta 1.0 \
  --gamma 0.8 \
  --tau 0.8 \
  --noise 5e-3 \
  --temperature 0.1 \
  --learning-rate 0.001 \
  --num-epochs 100
```

### Tune Base SOM-VAE

```bash
python main.py \
  --model somvae \
  --mode tune \
  --variant base \
  --dataset HAR \
  --seeds 42 43 44
```

### Tune Contrastive SOM-VAE

```bash
python main.py \
  --model somvae \
  --mode tune \
  --variant contrastive \
  --dataset PAMAP2 \
  --seeds 42 43 44
```

---

## SimCLR

SimCLR runs as a base training experiment.

```bash
python main.py \
  --model simclr \
  --mode train \
  --variant base \
  --dataset MHEALTH \
  --seeds 42 43 44 \
  --latent_dim 16 \
  --hidden_dim 16 \
  --num_states 10 \
  --noise 5e-3 \
  --temperature 0.1 \
  --learning-rate 0.001 \
  --num-epochs 100
```

---

## Important Arguments

### Model

```bash
--model somvae
--model catvae
--model simclr
```

### Mode

```bash
--mode train
--mode tune
```

Use `train` to run a fixed hyperparameter configuration.

Use `tune` to run a grid search over multiple hyperparameter configurations.

### Variant

```bash
--variant base
--variant contrastive
```

The contrastive variant uses data augmentation and a contrastive objective.

### Dataset

```bash
--dataset HAR
--dataset MHEALTH
--dataset PAMAP2
```

### Seeds

```bash
--seeds 42 43 44
```

Multiple seeds are recommended for stable reporting.

### Train/Validation Split

```bash
--ratio 0.8
```

The default train/validation split ratio is `0.8`.

---

## Reproducibility

For reproducible experiments, report the following information:

- dataset
- model
- variant
- mode
- random seeds
- train/validation split ratio
- model hyperparameters
- learning rate
- number of epochs
- noise level for contrastive experiments
- contrastive temperature for contrastive experiments

Example:

```bash
python main.py \
  --model catvae \
  --mode train \
  --variant contrastive \
  --dataset MHEALTH \
  --seeds 42 43 44 45 46 \
  --ratio 0.8 \
  --enc-out-dim 16 \
  --dec-out-dim 8 \
  --cat-dim 20 \
  --beta 1.0 \
  --gumbel-temperature 0.9 \
  --noise 5e-3 \
  --temperature 0.1 \
  --learning-rate 0.001 \
  --num-epochs 100
```

---

## Output

Each experiment returns a result dictionary from the corresponding runner function.

The result dictionary contains metadata such as:

```text
model
variant
mode
dataset
input_dim
train_ratio
seeds
device
results
```

## License

This project is released under the MIT License.

See `LICENSE` for details.
