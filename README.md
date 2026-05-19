# Patch-Level LPBF Microstructure Model

This folder contains a first-stage patch-level model for your constant-VED LPBF project.

The model uses one shared MLP backbone with two heads:

- classification head: predicts `coarse`, `fine`, or `boundary`
- regression head: predicts local grain size

## 1. Data Format

Prepare one CSV file where each row is one EBSD patch.

Required columns:

```text
run_id
patch_id
P
v
h
x_norm
y_norm
signed_distance_to_boundary
T_peak
cooling_rate
zone_label
local_grain_size
```

Label definition:

```text
0 = coarse
1 = fine
2 = boundary
```

The eight model inputs are:

```text
P, v, h, x_norm, y_norm, signed_distance_to_boundary, T_peak, cooling_rate
```

The two outputs are:

```text
zone_label
local_grain_size
```

## 2. Install Dependencies

Recommended environment:

```text
VS Code
Python 3.10 or 3.11
PyTorch
NumPy
```

Open this project folder in VS Code:

```text
C:\Users\yanhe\Documents\Codex\2026-05-18\new-chat
```

Then open the VS Code terminal and create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r patch_model/requirements.txt
```

In VS Code, select this Python interpreter:

```text
Ctrl + Shift + P
Python: Select Interpreter
.venv\Scripts\python.exe
```

## 3. Run With Synthetic Demo Data

Generate a fake dataset:

```powershell
python patch_model/make_synthetic_data.py --output patch_model/synthetic_patch_data.csv
```

Train and validate:

```powershell
python patch_model/train_patch_model.py --data patch_model/synthetic_patch_data.csv --output-dir patch_model/runs_demo
```

For a quick smoke test, use fewer epochs:

```powershell
python patch_model/train_patch_model.py --data patch_model/synthetic_patch_data.csv --epochs 100 --output-dir patch_model/quick_test
```

This runs:

- Leave-One-Run-Out cross validation
- a secondary 5-train / 2-test split using Run 2, 13, 14, 15, 16 for training and Run 17, 18 for testing

## 4. Run With Your Real Data

Fill `patch_model/patch_data_template.csv` with your own patch data, or save a new file such as:

```text
patch_model/my_patch_data.csv
```

Then run:

```powershell
python patch_model/train_patch_model.py --data patch_model/my_patch_data.csv --output-dir patch_model/runs_real
```

If you want to use the template file directly:

```powershell
python patch_model/train_patch_model.py --data patch_model/patch_data_template.csv --output-dir patch_model/runs_real
```

## 5. Outputs

The training script saves:

```text
metrics_leave_one_run_out.csv
metrics_fixed_split.csv
summary.json
fold_*/model.pt
```

Classification metrics:

```text
accuracy
macro_f1
confusion matrix
```

Regression metrics:

```text
MAE
RMSE
R2
```

Important: validation is grouped by `run_id`. Do not randomly split patches from the same run into both training and validation.

## 6. Model Structure

Inputs:

```text
P
v
h
x_norm
y_norm
signed_distance_to_boundary
T_peak
cooling_rate
```

Shared network:

```text
8 -> 32 -> 16
```

Classification head:

```text
16 -> 3
```

Classes:

```text
0 = coarse
1 = fine
2 = boundary
```

Regression head:

```text
16 -> 1
```

Regression output:

```text
local_grain_size
```

## 7. Training Settings

Optimizer:

```text
Adam
```

Default learning rate:

```text
1e-3
```

Default weight decay:

```text
1e-4
```

Default epochs:

```text
500
```

Default batch size:

```text
64
```

Default dropout:

```text
0.10
```

Loss function:

```text
total_loss = classification_loss + regression_weight * regression_loss
```

where:

```text
classification_loss = CrossEntropyLoss
regression_loss = MSELoss
regression_weight = 1.0
```

The classification task predicts:

```text
coarse / fine / boundary
```

The regression task predicts:

```text
local_grain_size
```

Feature normalization:

```text
Input features are standardized using the training set mean and standard deviation.
```

Regression target normalization:

```text
local_grain_size is standardized using the training set mean and standard deviation.
```

Validation strategy:

```text
Primary: Leave-One-Run-Out cross validation
Secondary: Run 2, 13, 14, 15, 16 for training; Run 17, 18 for testing
```

You can change the main training hyperparameters from the command line:

```powershell
python patch_model/train_patch_model.py ^
  --data patch_model/synthetic_patch_data.csv ^
  --epochs 1000 ^
  --batch-size 64 ^
  --lr 1e-3 ^
  --weight-decay 1e-4 ^
  --regression-weight 1.0 ^
  --dropout 0.10 ^
  --output-dir patch_model/runs_demo
```
