<div align="center">
    <table style="border: none; border-collapse: collapse;">
        <tr>
            <td align="center" width="20%" style="border: none;">
                <img src="https://i.ibb.co/yXKQmtZ/logo1.png" width="120" alt="University of Tehran Logo"/>
            </td>
            <td align="center" width="60%" style="border: none;">
                <h1>Deep Generative Models<br><small>Course Projects & Assignments</small></h1>
            </td>
            <td align="center" width="20%" style="border: none;">
                <img src="https://i.ibb.co/wLjqFkw/logo2.png" width="150" alt="ECE Faculty Logo"/>
            </td>
        </tr>
    </table>
</div>

---

<div align="center">

![Language](https://img.shields.io/badge/Language-Python%203.x-yellow)
![Course](https://img.shields.io/badge/Course-Deep%20Generative%20Models-blue)
![Semester](https://img.shields.io/badge/Semester-Fall%202025-orange)
![University](https://img.shields.io/badge/University-University%20of%20Tehran-red)

</div>

## Overview

This repository contains a comprehensive collection of assignments and projects implemented for the **Deep Generative Models** course. The objective of this repository is to demonstrate both the theoretical foundations and practical applications of state-of-the-art generative AI techniques. 

The projects range from building core architectures from scratch to fine-tuning large-scale pre-trained models, spanning across image generation, anomaly detection, image-to-image translation, and time-series modeling.

## Academic Context

* **Course:** Deep Generative Models
* **Institution:** University of Tehran, School of Electrical & Computer Engineering (ECE)
* **Semester:** Fall 2025 (1404-1405)
* **Level:** Undergraduate (B.Sc.) - 9th Semester
* 
## Repository Structure

This repository is organized into 8 standalone project directories. Each directory contains the implementation and analysis of a specific family of generative models:

* **[01-vae-and-beta-vae](./01-vae-and-beta-vae):** 
  Implementation of Standard VAE and $\beta$-VAE for disentangled representation learning (evaluated using Mutual Information Gap on dSprites).
* **[02-normalizing-flow-anomaly-detection](./02-normalizing-flow-anomaly-detection):** 
  Implementation of Masked Autoregressive Flow (MAF) utilizing MADE blocks for density estimation and unsupervised anomaly detection.
* **[03-cyclegan-image-translation](./03-cyclegan-image-translation):** 
  Implementation of CycleGAN (ResNet Generator + PatchGAN Discriminator) for unpaired image-to-image translation.
* **[04-energy-based-models](./04-energy-based-models):** 
  Designing and training CNN-based EBMs utilizing Contrastive Divergence and Langevin sampling for image generation and denoising.
* **[05-score-based-models](./05-score-based-models):** 
  Implementation of Noise Conditional Score Networks (NCSN) using annealed Langevin dynamics for conditional and unconditional generation.
* **[06-ddpm-ddim-from-scratch](./06-ddpm-ddim-from-scratch):** 
  From-scratch mathematical implementation of Forward/Reverse diffusion processes, comparing DDPM and DDIM sampling methodologies.
* **[07-stable-diffusion-dreambooth-lora](./07-stable-diffusion-dreambooth-lora):** 
  Fine-tuning Stable Diffusion (v1.5) using DreamBooth and LoRA for subject-driven generation (Woody from Toy Story), demonstrating the impact of Classifier-Free Guidance.
* **[08-flow-matching-time-series](./08-flow-matching-time-series):** 
  Applying Flow Matching via Continuous Normalizing Flows (CNFs) on a 1D U-Net to model and synthesize realistic financial time-series data (S&P 500).
  
## Tech Stack

The following libraries and tools were utilized across the projects:

| Category | Libraries & Tools |
| :--- | :--- |
| **Deep Learning Core** | `PyTorch`, `Torchvision`, `Torch.nn` |
| **Diffusion & GenAI** | `Diffusers` (HuggingFace), `PEFT` (LoRA) |
| **Data Processing** | `NumPy`, `Pandas`, `yfinance` |
| **Metrics & Evaluation** | `Scipy` (Sliced Wasserstein Distance), PCA |
| **Visualization** | `Matplotlib`, `Tqdm` |

## Getting Started

To access the code and notebooks, clone the repository using the following command:

```bash
git clone [https://github.com/YOUR_USERNAME/dgm-course-projects.git](https://github.com/Armin-ghasemi/dgm-course-projects.git)
cd dgm-course-projects
