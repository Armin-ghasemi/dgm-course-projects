# Energy-Based Modeling with Langevin Dynamics

![Course](https://img.shields.io/badge/Course-Deep%20Generative%20Models-blue)
![University](https://img.shields.io/badge/University-University%20of%20Tehran-red)
![Assignment](https://img.shields.io/badge/Assignment-HW3%20--%20Question%201-green)

This project implements an **Energy-Based Model (EBM)** for generative modeling on the MNIST dataset. The model learns an energy landscape in which images from the data distribution are assigned lower energy values. **Langevin Dynamics** is then used to sample from and explore the learned energy landscape.

## Project Goals

1. **Energy-Based Modeling:** Implement an Energy-Based Model that assigns a scalar energy to each MNIST image.
2. **Energy Landscape Learning:** Train the model so that real MNIST images receive lower energy than generated samples.
3. **Langevin Sampling:** Use Langevin Dynamics to generate samples by following the gradient of the learned energy function.
4. **Image Generation:** Generate MNIST-like samples starting from random inputs.
5. **Image Refinement and Denoising:** Evaluate the learned energy landscape by refining real images and denoising corrupted images.

## Task 1: Energy-Based Model

The experiment uses the **MNIST** dataset of handwritten digits. Images are converted to tensors with pixel values in the range `[0, 1]` and loaded using PyTorch `DataLoader`s with a batch size of 64.

The goal is to learn an energy function `Eθ(x)` that assigns a scalar energy to each input image.

Unlike a classifier, the model does not predict digit labels. Instead, it learns an energy landscape in which samples from the data distribution are encouraged to have lower energy.

The corresponding energy-based distribution can be written as:

`pθ(x) ∝ exp(-Eθ(x))`

where `Eθ(x)` is the learned energy function and the proportionality constant corresponds to the normalization term.

### Energy Model Architecture

The energy function is implemented using a convolutional neural network. The network progressively extracts higher-level features through convolutional layers and maps the resulting representation to a single scalar energy value.

```text
Input image
    ↓
Convolutional layers
    ↓
Feature representation
    ↓
Scalar energy
```

The model is used both during training and during Langevin sampling.

### Langevin Dynamics

Langevin Dynamics is used to refine input images according to the learned energy landscape.

The update rule is:

`x(t+1) = x(t) - ε ∇x Eθ(x(t)) + √(2ε) z(t)`

where `ε` is the step size, `z(t)` is Gaussian noise, and `∇x Eθ(x)` is the gradient of the energy with respect to the input image.

A key part of the sampling process is computing the gradient with respect to the **input image**:

`∇x Eθ(x)`

This allows the image itself to be progressively refined during sampling.

### EBM Training

The Energy-Based Model is trained by contrasting real MNIST images with samples obtained from Langevin Dynamics.

For a real sample `x_real` and a generated sample `x_fake`, the main energy objective is:

`L_energy = Eθ(x_real) - Eθ(x_fake)`

An additional regularization term is used to prevent the energy values from growing without bound:

`L_reg = λ [Eθ(x_real)² + Eθ(x_fake)²]`

The final training objective is:

`L = L_energy + L_reg`

The model is trained for **10 epochs** with a batch size of **64** and an initial learning rate of `1e-3`. The regularization coefficient is set to `λ = 0.01`. Langevin sampling uses a step size of `3.0`, 100 sampling steps, and a noise scale of `0.005`.

### Training Progress

During training, samples are generated from random inputs at different epochs to visualize how the learned energy landscape evolves.

The final generation results after the 10th epoch are shown below.

![Epoch 10 Generations](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/04-energy-based-models/assets/showcase/epoch10_generations.png)

## Sampling and Denoising Experiments

After training, the learned energy landscape is evaluated through several Langevin-based experiments.

### Refinement from Real Images

The first experiment starts from real MNIST images and applies Langevin Dynamics. This evaluates whether the learned energy landscape can refine an already valid sample while preserving its main structure.

The original images are compared with the outputs produced by Langevin Dynamics.

### Generation from Random Noise

The second experiment starts from randomly initialized images rather than real MNIST samples.

The sampling process gradually moves the random inputs toward low-energy regions:

```text
x₀ ~ Uniform(0, 1)
      ↓
     x₁
      ↓
     ...
      ↓
     xT
```

The final generated samples are compared with real MNIST images.

![Generated and Real Samples](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/04-energy-based-models/assets/showcase/generated_vs_real.png)

### Denoising with Langevin Dynamics

The final experiment evaluates the model as a denoising mechanism.

Gaussian noise is added to real MNIST images:

`x_noisy = x + σz`

where `σ` controls the noise level and `z` is Gaussian noise.

Three noise levels are evaluated:

`σ ∈ {0.2, 0.4, 0.6}`

The model attempts to move the corrupted images toward regions corresponding to cleaner and more likely MNIST samples.

![Denoising with Langevin Dynamics](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/04-energy-based-models/assets/showcase/denoising_sigma_06.png)

## Models & Analysis

| Model                  | Training Objective               | Sampling Method   | Key Analysis                                                                                                                       |
| :--------------------- | :------------------------------- | :---------------- | :--------------------------------------------------------------------------------------------------------------------------------- |
| **Energy-Based Model** | Energy contrast + regularization | Langevin Dynamics | Learns an energy landscape that assigns lower energy to real MNIST samples and uses input gradients for generation and refinement. |

The notebook evaluates the learned model from three perspectives: **generation from random inputs, refinement of real images, and denoising of corrupted images**.

## Sample Results

The final results demonstrate the use of the learned energy landscape for both generation and image refinement. Samples generated from random inputs are compared with real MNIST images, while denoising experiments evaluate the model under different levels of Gaussian corruption.

## File Structure

```text
<FOLDER_NAME>/
├── ebm_mnist.ipynb
├── README.md
└── assets/
    └── showcase/
        ├── epoch10_generations.png
        ├── generated_vs_real.png
        └── denoising_sigma_06.png
```

* `ebm_mnist.ipynb`: The main notebook containing the EBM implementation, Langevin Dynamics, training procedure, sampling experiments, and denoising analysis.
* `assets/showcase/`: Selected visualizations from the notebook used in this README.

## How to Run

1. Install the required packages:

```bash
pip install torch torchvision numpy matplotlib torchinfo
```

2. Open `ebm_mnist.ipynb` in Jupyter Notebook or Google Colab.
3. Run the notebook cells to download the MNIST dataset, train the Energy-Based Model, and perform the sampling and denoising experiments.
4. GPU acceleration is recommended for training and Langevin sampling.
