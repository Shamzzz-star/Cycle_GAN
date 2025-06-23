# 🌀 CycleGAN for Image-to-Image Translation

This project implements a CycleGAN for unpaired image-to-image translation, focusing on transforming horse images to zebra images and vice versa. It includes:

- `cyclegancode.py`: Training script.
- `eval.py`: Evaluation using FCN-ResNet101 for semantic confidence scoring.
- Support for generating and saving translated images and model checkpoints.

---

## 🗂️ Project Structure

```
.
├── cyclegancode.py           # Training script for CycleGAN
├── eval.py                   # Evaluation script using FCN-ResNet101
├── outputs/                  # Generated images during training
├── result/
│   └── 100eppoch/
│       └── geno.pth.tar      # Pretrained Generator checkpoint
└── datasets/
    └── apple2orange/         # Unpaired dataset
        ├── trainA/           # Domain A images (e.g., horses)
        ├── trainB/           # Domain B images (e.g., zebras)
        └── testA/            # Test images from domain A
```

---

## 🚀 Getting Started

### 1. Training the Model

To train the CycleGAN:

```bash
python cyclegancode.py
```

This script:
- Trains Generator and Discriminator networks.
- Saves output images to `outputs/` every 50 iterations.
- Saves model checkpoints in `result/` every 5 epochs.

> ✅ Make sure dataset paths are correct in the script.

---

### 2. Evaluation

Evaluate the quality of translated images using FCN-ResNet101:

```bash
python eval.py
```

The evaluation script:
- Loads the pretrained generator (`geno.pth.tar`).
- Translates `testA` images to the target domain.
- Computes semantic segmentation confidence via FCN.
- Prints the average semantic confidence score.

> You can modify the paths to the checkpoint and test folder in the `main()` function of `eval.py`.

---

## 🧠 Model Architecture

### Generator
- Inspired by U-Net.
- Includes downsampling, residual blocks, and upsampling layers.
- Performs domain A → B and domain B → A translations.

### Discriminator
- PatchGAN architecture.
- Classifies patches as real or fake for finer-level realism.

---

## ✅ Requirements

- Python 3.8+
- PyTorch
- torchvision
- albumentations
- tqdm
- numpy
- Pillow

Install with:

```bash
pip install -r requirements.txt
```



---

## 📊 Output Examples

- `    visuals_z

/vis_n02381461_670.png` — Fake zebras generated from horses.
- `outputs/fake_horse_*.png` — Fake horses generated from zebras.
- `eval.py` prints: **"Average confidence score: 0.87"** (example output).

---

## 📎 Notes

- CycleGAN uses:
  - **Cycle Consistency Loss** (to preserve structure).
  - **Identity Loss** (to regularize generator).
- Trained for 120 epochs by default.
- CUDA recommended for faster training.

---


