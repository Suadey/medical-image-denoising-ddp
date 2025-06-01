# medical-image-denoising-ddp

# Distributed Deep Learning for Medical Image Denoising with Data Obfuscation

This repository contains the official PyTorch implementation of the paper:
**"Distributed Deep Learning for Medical Image Denoising with Data Obfuscation"**  
by Sulaimon Oyeniyi Adebayo and Ayaz H. Khan

## Project Summary

This work explores distributed deep learning using U-Net and U-Net++ for denoising chest X-rays from the NIH ChestX-ray14 dataset. The noisy images simulate data obfuscation using additive Gaussian noise, with training optimized using:
- PyTorch’s DistributedDataParallel (DDP)
- Automatic Mixed Precision (AMP)

## Dataset: NIH Chest X-ray14 (Kaggle)

We use a subset of the [NIH Chest X-ray14 Dataset on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data), specifically:

- **15,000 images** from the first two folders: `images_001` and `images_002`

### 🔗 Download Instructions

1. Go to the Kaggle dataset page: [NIH Chest X-ray14 Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data)
2. Download:
   - `images_001/`
   - `images_002/`

## Reproducibility
All code and trained model are provided in the respective folders.
All the training and testing  for 1 GPU and 2 GPU (DP) can be done right inside the notebook.

For Optimized Multi-GPU (DDP + AMP), it needs to be run from the terminal with the following

```bash
torchrun --nproc_per_node=2 train_ddp_unetpp.py
```

## If you find the repository useful, kindly cite the our paper:

```bibtex
@article{adebayo2025distributed,
  title={Distributed Deep Learning for Medical Image Denoising with Data Obfuscation},
  author={Adebayo, Sulaimon Oyeniyi and Khan, Ayaz H.},
  journal={arXiv preprint arXiv:2505.xxxxx},
  year={2025}
}
```


