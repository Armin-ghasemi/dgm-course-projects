# Noise-Conditional Score Network on MNIST

![Course](https://img.shields.io/badge/Course-Deep%20Generative%20Models-blue)
![University](https://img.shields.io/badge/University-University%20of%20Tehran-red)
![Assignment](https://img.shields.io/badge/Assignment-HW3%20--%20Question%202-green)

This project implements a **Noise-Conditional Score Network (NCSN)** for generative modeling on the MNIST dataset. The model learns the score function of noise-perturbed data distributions at multiple noise levels and uses **Annealed Langevin Dynamics** to generate samples.

A conditional version is also implemented, allowing the model to generate samples corresponding to specific MNIST digit classes.

## Project Goals

1. **Score Matching:** Train a neural network to estimate the score of noisy MNIST distributions.
2. **Noise Conditioning:** Condition the network on multiple noise levels to model distributions with different levels of corruption.
3. **NCSN Sampling:** Generate samples using Annealed Langevin Dynamics.
4. **Conditional Generation:** Extend the model with class conditioning to generate specific digit classes.
5. **Sample Evaluation:** Visualize the denoising process and final generated samples.

## Task 1: Noise-Conditional Score Network

The first part implements a Noise-Conditional Score Network for MNIST.

Instead of directly modeling the probability density, the network learns its score function:

`score(x) = ∇x log p(x)`

The score indicates the direction in which the input should move to reach regions of higher probability.

### Noise Perturbation

A set of **50 noise levels** is used, ranging geometrically from:

`σ_max = 30`

to:

`σ_min = 0.01`

For an original image `x`, Gaussian noise is added according to:

`x̃ = x + σz`

where `z` is sampled from a standard Gaussian distribution.

The network receives both the noisy image and its corresponding noise level and learns to estimate the score of the perturbed distribution.

### Noise-Level Embedding

The noise level is represented using a **Gaussian Fourier feature embedding**.

This embedding provides the network with a continuous representation of the noise level and allows the same model to operate across the full range of noise scales.

The embedded noise information is incorporated into the network through **FiLM conditioning**.

### Score Matching Objective

For the perturbed image `x̃ = x + σz`, the target score is:

`target = -z / σ`

The network is trained to predict this target score:

`scoreθ(x̃, σ) ≈ -z / σ`

This allows the model to learn the score field for each noise level without explicitly estimating the data density.

### NCSN Architecture

The score network follows a U-Net-like architecture with:

* Convolutional layers for feature extraction
* Residual blocks
* Downsampling and upsampling layers
* Skip connections
* Gaussian Fourier noise-level embeddings
* FiLM-based conditioning

The overall structure can be summarized as:

```text
Noisy image + Noise level
          ↓
   Noise embedding
          ↓
   U-Net-like network
          ↓
    Predicted score
```

### Training

The model is trained using noisy versions of MNIST images at randomly selected noise levels.

The training process minimizes the difference between the predicted score and the analytical score-matching target.

The training loss is monitored throughout the training process.

![NCSN Training Loss](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/05-score-based-models/assets/showcase/ncsn_training_loss.png)

## Annealed Langevin Dynamics

After training, samples are generated using **Annealed Langevin Dynamics**.

Instead of using a single noise level, sampling starts from a high-noise distribution and gradually moves toward lower noise levels.

At each noise level, the sample is updated using the learned score function.

A simplified update can be written as:

`x(t+1) = x(t) + α scoreθ(x(t), σ) + √(2α) z(t)`

where `α` is the Langevin step size and `z(t)` is Gaussian noise.

The sampling procedure follows the noise schedule from the largest noise level to the smallest:

```text
High noise
   ↓
σ₁
   ↓
σ₂
   ↓
...
   ↓
σ₅₀
   ↓
Low noise
   ↓
Generated sample
```

This gradual transition allows the model to first capture the broad structure of the data and then refine the generated samples at lower noise levels.

### Denoising Process

The intermediate states of the sampling process are visualized to show how noisy inputs gradually develop into recognizable MNIST digits.

![NCSN Denoising Process](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/05-score-based-models/assets/showcase/ncsn_denoising.png)

## Task 2: Conditional NCSN

The second part extends the NCSN to support **class-conditional generation**.

MNIST contains 10 digit classes, from `0` to `9`. A learnable class embedding is introduced so that the score network receives both the noise-level information and the desired digit class.

The conditional score function can be represented as:

`scoreθ(x, σ, y)`

where:

* `x` is the noisy image
* `σ` is the noise level
* `y` is the target digit class

The class embedding is incorporated into the network together with the noise-level embedding.

### Conditional Sampling

During sampling, a target class is specified for each generated sample.

The model then performs Annealed Langevin Dynamics while conditioning the score predictions on the selected class.

The final experiment generates **16 samples for each of the 10 MNIST classes**, producing an ordered grid of 160 generated images.

![Conditional Samples](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/05-score-based-models/assets/showcase/conditional_samples.png)

## Models & Analysis

| Model                | Conditioning              | Sampling Method            | Key Analysis                                                            |
| :------------------- | :------------------------ | :------------------------- | :---------------------------------------------------------------------- |
| **NCSN**             | Noise level               | Annealed Langevin Dynamics | Learns the score field of multiple noise-perturbed MNIST distributions. |
| **Conditional NCSN** | Noise level + digit class | Annealed Langevin Dynamics | Generates MNIST samples conditioned on a specified digit class.         |

The two models demonstrate how score-based generative modeling can be extended from unconditional generation to class-conditional generation.

## Sample Results

The experiments visualize both the intermediate denoising process and the final generated samples.

The unconditional NCSN demonstrates progressive refinement from highly noisy states toward recognizable MNIST-like images. The Conditional NCSN further allows the generated samples to be controlled by specifying the desired digit class.

## File Structure

```text
<FOLDER_NAME>/
├── NCSN_MNIST.ipynb
├── README.md
└── assets/
    └── showcase/
        ├── ncsn_training_loss.png
        ├── ncsn_denoising.png
        └── conditional_samples.png
```

* `NCSN_MNIST.ipynb`: The main notebook containing the NCSN and Conditional NCSN implementations, training procedure, and sampling experiments.
* `assets/showcase/`: Selected visualizations from the notebook used in this README.

## How to Run

1. Install the required packages:

```bash
pip install torch torchvision numpy matplotlib
```

2. Open `NCSN_MNIST.ipynb` in Jupyter Notebook or Google Colab.
3. Run the notebook cells to download the MNIST dataset, train the NCSN models, and perform Annealed Langevin Dynamics sampling.
4. GPU acceleration is recommended for training and sampling.
