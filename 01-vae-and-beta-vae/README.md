# Variational Autoencoder and Beta-VAE on dSprites

![University](https://img.shields.io/badge/University-University%20of%20Tehran-red)
![Course](https://img.shields.io/badge/Course-Deep%20Generative%20Models-blue)
![Assignment](https://img.shields.io/badge/Assignment-HW1--Q2-green)

This project implements and analyzes a **Variational Autoencoder (VAE)** and **Beta-VAE** models on the **dSprites** dataset. The main objective is to study how the strength of latent-space regularization affects reconstruction quality and the learned latent representations.

Three models are evaluated with beta values of **1, 3, and 18**. Their reconstruction and generation results are examined together with **Mutual Information Gap (MIG)** scores and **PCA-based latent-space visualizations**.

## Project Goals

1. Implement a convolutional **Variational Autoencoder** for the dSprites dataset.
2. Extend the VAE to a **Beta-VAE** by varying the weight of the KL-divergence term.
3. Compare reconstruction and generation results for beta = 1, 3, and 18.
4. Evaluate the learned representations using **Mutual Information Gap (MIG)**.
5. Visualize the learned latent spaces using **Principal Component Analysis (PCA)**.

## Task 1: Variational Autoencoder

The VAE consists of a convolutional encoder and decoder connected through a **10-dimensional latent space**.

The encoder maps each input image to the mean and log-variance of a Gaussian distribution. A latent vector is then sampled using the reparameterization trick and passed to the decoder.

The overall pipeline is:

```text
Input Image
    |
    v
Encoder
    |
    v
Mean and Log-Variance
    |
    v
Latent Vector
    |
    v
Decoder
    |
    v
Reconstructed Image
```

The VAE loss consists of two components:

```text
VAE Loss = Reconstruction Loss + KL Divergence
```

The reconstruction loss measures how closely the reconstructed image matches the input, while the KL-divergence term regularizes the latent distribution toward the prior distribution.

### Dataset

The **dSprites** dataset is a synthetic dataset designed for studying disentangled representations. Each image is generated from known factors including:

* Shape
* Scale
* Orientation
* Position X
* Position Y

The dataset contains **737,280 images** with a resolution of **64 × 64**. The images are stored as `uint8` arrays for efficient memory usage, while the corresponding ground-truth latent factors are preserved for later evaluation.

### Training Configuration

| Parameter               |             Value |
| ----------------------- | ----------------: |
| Image size              |           64 × 64 |
| Latent dimension        |                10 |
| Batch size              |               128 |
| Optimizer               |              Adam |
| Initial learning rate   |             0.001 |
| Maximum epochs          |               100 |
| LR scheduler            | ReduceLROnPlateau |
| Scheduler patience      |                 8 |
| Scheduler factor        |               0.5 |
| Early stopping patience |                15 |

## Task 2: Beta-VAE Experiments

The standard VAE is treated as the **beta = 1** case. Two additional models are trained using **beta = 3** and **beta = 18**, while keeping the main VAE architecture unchanged.

The Beta-VAE loss can be written conceptually as:

```text
Beta-VAE Loss = Reconstruction Loss + beta × KL Divergence
```

Increasing beta gives more weight to the KL-divergence term and therefore imposes stronger regularization on the latent representation.

### beta = 1

**Original**

![Original images for beta=1](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/01-vae-and-beta-vae/assets/showcase/beta1_original.png)

**Reconstruction**

![Reconstructed images for beta=1](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/01-vae-and-beta-vae/assets/showcase/beta1_reconstruction.png)

**Generated Samples**

![Generated images for beta=1](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/01-vae-and-beta-vae/assets/showcase/beta1_generated.png)

### beta = 3

**Original**

![Original images for beta=3](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/01-vae-and-beta-vae/assets/showcase/beta3_original.png)

**Reconstruction**

![Reconstructed images for beta=3](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/01-vae-and-beta-vae/assets/showcase/beta3_reconstruction.png)

**Generated Samples**

![Generated images for beta=3](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/01-vae-and-beta-vae/assets/showcase/beta3_generated.png)

### beta = 18

**Original**

![Original images for beta=18](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/01-vae-and-beta-vae/assets/showcase/beta18_original.png)

**Reconstruction**

![Reconstructed images for beta=18](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/01-vae-and-beta-vae/assets/showcase/beta18_reconstruction.png)

**Generated Samples**

![Generated images for beta=18](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/01-vae-and-beta-vae/assets/showcase/beta18_generated.png)

For **beta = 18**, the stronger KL regularization substantially limits the information retained in the latent representation. This results in noticeably weaker reconstruction and generation quality compared with the lower-beta models. The same behavior is also reflected in the very low MIG score obtained by this model.

## Disentanglement Evaluation with MIG

Visual inspection alone is not sufficient to evaluate disentanglement. Since dSprites provides ground-truth generative factors, the relationship between these factors and the learned latent dimensions can be evaluated quantitatively.

The **Mutual Information Gap (MIG)** is used for this purpose.

For each ground-truth factor, the mutual information between that factor and every latent dimension is calculated. The highest and second-highest mutual information values are then compared, and the normalized gaps are averaged across the ground-truth factors.

The evaluation uses **50,000 samples** and considers the following factors:

* Shape
* Scale
* Orientation
* Position X
* Position Y

### MIG Results

| Model    | Beta |        MIG |
| -------- | ---: | ---------: |
| VAE      |    1 |     0.0316 |
| Beta-VAE |    3 | **0.0860** |
| Beta-VAE |   18 |     0.0026 |

The **beta = 3** model achieves the highest MIG score among the three experiments. In contrast, the very small MIG score for **beta = 18** is consistent with the near-zero mutual information observed between the latent dimensions and the ground-truth factors.

## Latent Space Visualization with PCA

To further inspect the learned latent representations, the 10-dimensional latent vectors are projected into two dimensions using **Principal Component Analysis (PCA)**.

PCA is used only as a visualization method and does not directly measure disentanglement. The same procedure is applied to all three models to allow their latent-space structures to be visually compared.

### VAE (beta = 1)

![PCA visualization for beta=1](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/01-vae-and-beta-vae/assets/showcase/beta1_pca.png)

### Beta-VAE (beta = 3)

![PCA visualization for beta=3](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/01-vae-and-beta-vae/assets/showcase/beta3_pca.png)

### Beta-VAE (beta = 18)

![PCA visualization for beta=18](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/01-vae-and-beta-vae/assets/showcase/beta18_pca.png)

## File Structure

```text
.
├── README.md
├── vae_beta_vae.ipynb
└── assets/
    └── showcase/
        ├── beta1_original.png
        ├── beta1_reconstruction.png
        ├── beta1_generated.png
        ├── beta3_original.png
        ├── beta3_reconstruction.png
        ├── beta3_generated.png
        ├── beta18_original.png
        ├── beta18_reconstruction.png
        ├── beta18_generated.png
        ├── beta1_pca.png
        ├── beta3_pca.png
        └── beta18_pca.png
```

## How to Run

1. Clone the repository.
2. Install the required Python dependencies.
3. Open `vae_beta_vae.ipynb` using Jupyter Notebook, JupyterLab, or Google Colab.
4. Download and prepare the dSprites dataset as described in the notebook.
5. Run the notebook cells sequentially to train the models and reproduce the reported results and visualizations.
