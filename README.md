# CycleGAN for Unpaired Image Translation

This project implements **Cycle-Consistent Adversarial Networks (CycleGAN)** for unpaired image-to-image translation tasks. It is inspired by the original [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) implementation by Jun-Yan Zhu et al.

We applied CycleGAN to perform style translation between:
- 🐴 Horse ↔ 🦓 Zebra
- 🍎 Apple ↔ 🍊 Orange

All outputs are available in the [`outputs/`](./outputs) folder.

---

## ✨ Example Results

### Horse ↔ Zebra

<table>
<tr>
<td><b>Input (Horse)</b></td>
<td><b>Fake Zebra</b></td>
<td><b>Reconstructed Horse</b></td>
</tr>
<tr>
<td><img src="./outputs/horse2zebra/input1.jpg" width="200"/></td>
<td><img src="./outputs/horse2zebra/fake1.jpg" width="200"/></td>
<td><img src="./outputs/horse2zebra/reconstructed1.jpg" width="200"/></td>
</tr>
</table>

---

### Apple ↔ Orange

<table>
<tr>
<td><b>Input (Apple)</b></td>
<td><b>Fake Orange</b></td>
<td><b>Reconstructed Apple</b></td>
</tr>
<tr>
<td><img src="./outputs/apple2orange/input1.jpg" width="200"/></td>
<td><img src="./outputs/apple2orange/fake1.jpg" width="200"/></td>
<td><img src="./outputs/apple2orange/reconstructed1.jpg" width="200"/></td>
</tr>
</table>

---

## 📁 Dataset

We used unpaired image datasets from the [CycleGAN dataset collection](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/):
- `horse2zebra`
- `apple2orange`

You can download datasets using:
```bash
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
bash ./datasets/download_cyclegan_dataset.sh apple2orange
