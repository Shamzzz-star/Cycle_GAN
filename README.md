# CycleGAN for Unpaired Image Translation

This project implements **Cycle-Consistent Adversarial Networks (CycleGAN)** for unpaired image-to-image translation, inspired by [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Our work focuses on translating images between:
- 🐴 Horse ↔ 🦓 Zebra
- 🍎 Apple ↔ 🍊 Orange

---

## ✨ Sample Outputs

### Zebra Domain Output
![](./outputs/visuals_z/vis_n02381461_1000.png)
![](./outputs/visuals_z/vis_n02381461_1030.png)

### Orange Domain Output
![](./outputs/visuals_o/vis_n07740461_10311.png)

---

## 📁 Dataset

We used datasets from the [CycleGAN dataset collection](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/):
- `horse2zebra`
- `apple2orange`

To download:

```bash
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
bash ./datasets/download_cyclegan_dataset.sh apple2orange
