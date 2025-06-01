# medical-image-denoising-ddp

# Distributed Deep Learning for Medical Image Denoising with Data Obfuscation

This repository contains the official PyTorch implementation of the paper:
**"Distributed Deep Learning for Medical Image Denoising with Data Obfuscation"**  
by Sulaimon Oyeniyi Adebayo and Ayaz H. Khan

## 📌 Project Summary

This work explores distributed deep learning using U-Net and U-Net++ for denoising chest X-rays from the NIH ChestX-ray14 dataset. The noisy images simulate data obfuscation using additive Gaussian noise, with training optimized using:
- PyTorch’s DistributedDataParallel (DDP)
- Automatic Mixed Precision (AMP)

## 🗂 Folder Structure

All the training and testing for 1 GPU and 2 GPU DP can be done right inside the notebook.

For Optimized Multi-GPU (DDP + AMP), it needs to be run from the terminal with the following

```bash
torchrun --nproc_per_node=2 train_ddp_unetpp.py # 
