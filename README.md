# Road Sign Occlusion Detection with a Custom CNN

This project trains a compact convolutional neural network to predict two things from each traffic-sign image: its GTSRB class (43 classes) and whether synthetic occlusion is present. The shared CNN trunk feeds separate classification and binary occlusion heads.

## Dataset and augmentation

The code uses the public German Traffic Sign Recognition Benchmark (GTSRB), downloaded by `torchvision.datasets.GTSRB` on first run. Training images receive on-the-fly rectangular overlays with earth, foliage, or grime-like colors, with varied size, position, blending, and count. An occlusion label of 1 is returned whenever an overlay is applied. The test split is consistently occluded to measure the requested detection behavior. This is a reproducible synthetic proxy; it does not claim to represent all real-world obstruction types.

## Setup

Python 3.10 or newer is recommended. From this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train

```powershell
.\.venv\Scripts\python.exe train.py --epochs 15 --batch-size 128
```

The dataset is downloaded into `data/`. The best checkpoint by occlusion F1 is written to `checkpoints/road_sign_cnn.pt`. Use `--data-dir`, `--output`, `--epochs`, `--batch-size`, and `--workers` to adjust the run. Training automatically uses CUDA when available.

## Evaluate

```powershell
.\.venv\Scripts\python.exe evaluate.py
```

Evaluation prints sign accuracy and occlusion precision, recall, and F1. Precision measures how many occlusion alerts are correct; recall measures how many occluded signs were caught. The default decision threshold is 0.5 (logit >= 0).

## Model and objective

Four convolutional blocks with batch normalization and max pooling extract image features. Global average pooling and a compact shared dense representation feed a 43-class softmax-logit head and a one-logit binary occlusion head. Training minimizes cross-entropy for sign class plus binary cross-entropy-with-logits for occlusion.

## Limitations

Synthetic overlays teach the model the augmentation patterns used here and may not generalize to natural leaves, shadows, glare, snow, physical damage, or unfamiliar camera conditions. A real deployment would require representative labeled occlusion data, calibration, and safety validation. This coursework model is not a driving safety system.
