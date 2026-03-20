# Human Pose Estimation with Custom Hourglass Network (PyTorch)

This project implements a **2D human pose estimation pipeline** using a **custom Hourglass neural network** trained on the **MPII Human Pose Dataset**.

The model predicts **body keypoints (joints)** from RGB images using **heatmap regression**, enabling structured understanding of human pose in real-world scenes.

---

## Highlights

- Custom implementation of **Hourglass architecture** in PyTorch
- End-to-end pipeline: **data preprocessing → training → inference → visualization**
- Training on **MPII dataset** with custom heatmap generation
- Focus on **understanding model behavior**, not just using pretrained solutions
- Clean, modular codebase for experimentation

---

## Overview

Human pose estimation is a key problem in computer vision, with applications in:

- robotics and human-robot interaction
- motion tracking and activity recognition
- sports analytics
- augmented reality

This project approaches pose estimation as a **supervised learning problem**, where the network learns to predict **per-joint heatmaps** representing the probability of joint locations.

---

## Model Architecture

The model is based on a **custom Hourglass network**, designed to capture both:

- **global spatial relationships** (body structure)
- **fine-grained localization** (exact joint positions)

Key characteristics:

- Repeated **downsampling and upsampling**
- **Skip connections** for preserving spatial detail
- Output: **N heatmaps**, one per body joint

The architecture was implemented from scratch to better understand pose estimation internals.

---

## Dataset

The model is trained on the **MPII Human Pose Dataset**, which includes:

- diverse real-world human poses
- multiple activities and viewpoints
- challenging conditions (occlusion, scale variation)

Custom preprocessing includes:

- parsing annotation format
- normalizing joint coordinates
- generating Gaussian heatmaps for supervision

---

## Training Pipeline

The project includes a full training workflow:

1. Load and preprocess MPII dataset
2. Generate ground-truth heatmaps for each joint
3. Train Hourglass model using heatmap regression loss
4. Validate predictions on unseen samples
5. Visualize predicted keypoints and heatmaps

---

## Inference & Visualization

The trained model can:

- predict body keypoints from a single image
- generate interpretable joint heatmaps
- overlay skeleton predictions on input images

This makes it easy to debug and qualitatively evaluate performance.

---

## Tech Stack

- **Python**
- **PyTorch**
- **OpenCV**
- **NumPy**
- **Matplotlib**

---

## Project Structure

```bash
.
├── datasets.py         # MPII preprocessing and loading
├── models.py           # Hourglass network implementation
├── train.py            # Training script
├── demo.py             # Run model on new images
├── losses.py           # Define loss function
├── util*.py            # Util scripts
└── README.md
