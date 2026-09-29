# Score-Based Generative Modeling with NCSN

![Course](https://img.shields.io/badge/Course-Deep%20Generative%20Models-blue)
![University](https://img.shields.io/badge/University-University%20of%20Tehran-red)
![Assignment](https://img.shields.io/badge/Assignment-HW3%20--%20Question%202-green)

This project implements **Noise-Conditional Score Networks (NCSN)** for score-based generative modeling on the MNIST dataset. The assignment consists of two parts: first, training an NCSN to estimate the score of noise-perturbed image distributions, and then extending the model with class conditioning for controlled generation of specific MNIST digits.

## Project Goals

1. **Score Estimation:** Implement a Noise-Conditional Score Network to estimate the score function at different noise levels.
2. **Noise Conditioning:** Incorporate noise-level information using Gaussian Fourier embeddings and FiLM-based conditioning.
3. **Score-Based Generation:** Generate MNIST samples using annealed Langevin dynamics.
4. **Conditional Generation:** Extend the NCSN with digit-class conditioning to generate samples from specified MNIST classes.
5. **Generation Analysis:** Analyze the training process, denoising behavior, and generated samples.

## Task 1: Noise-Conditional Score Network

The first part implements a **Noise-Conditional Score Network (NCSN)** for score-based generative modeling on MNIST.

Instead of directly modeling the data distribution, the network estimates the score of a noise-perturbed distribution:

$$
s_\theta(x,\sigma) \approx \nabla_x \log p_\sigma(x)
$$

### Noise Levels

The model uses **50 noise levels**, ranging from $\sigma_{\max}=30$ to $\sigma_{\min}=0.01$ using a geometric progression.

For each training sample, Gaussian noise is added according to a selected noise level:

$$
\tilde{x}=x+\sigma z,
\qquad
z\sim\mathcal{N}(0,I)
$$

### Noise Conditioning

The noise level is encoded using a **Gaussian Fourier embedding** and incorporated into the network through **FiLM-based conditioning**.

### Score Matching Objective

For the Gaussian perturbation process, the target score is:

$$
\nabla_{\tilde{x}}\log q_\sigma(\tilde{x}|x)
=
-\frac{z}{\sigma}
$$

The network is trained using the corresponding weighted denoising score-matching objective.

### Network Architecture

The NCSN uses a **U-Net-like architecture** with residual blocks, downsampling and upsampling paths, skip connections, and noise-level conditioning. The network outputs a score map with the same spatial dimensions as the input image.

### Training and Sampling

The model is trained on MNIST and samples are generated using **annealed Langevin dynamics**, progressively transforming noisy samples into structured image samples across the sequence of noise levels.

![NCSN Training Loss](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/05-score-based-models/assets/showcase/ncsn_training_loss.png)

The denoising process is visualized below:

![NCSN Denoising Process](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/05-score-based-models/assets/showcase/ncsn_denoising.png)

## Task 2: Conditional NCSN

The second part extends the NCSN to support **class-conditional generation**.

The conditional model estimates:

$$
s_\theta(x,\sigma,y)
$$

where $y$ represents the desired MNIST digit class.

### Class Conditioning

A learnable embedding is used for the **10 MNIST digit classes**. The class embedding is combined with the noise-level embedding and incorporated into the residual blocks through the conditioning mechanism.

```text
Noisy image
    +
Noise level
    +
Target digit class
    ↓
Conditional NCSN
    ↓
Conditioned score map
```

### Conditional Sampling

During sampling, a target digit class is selected and kept fixed while **annealed Langevin dynamics** progressively refines the generated sample.

Multiple samples are generated for each digit class. The final visualization contains generated samples for all ten MNIST classes.

## Models & Analysis

| Model                | Conditioning              | Sampling Method                        | Key Analysis                                                                                                    |
| :------------------- | :------------------------ | :------------------------------------- | :-------------------------------------------------------------------------------------------------------------- |
| **NCSN**             | Noise level               | Annealed Langevin Dynamics             | Estimates the score of noise-perturbed MNIST distributions and generates samples through progressive denoising. |
| **Conditional NCSN** | Noise level + digit class | Conditional Annealed Langevin Dynamics | Generates samples guided by a specified MNIST digit class.                                                      |

The final conditional generation produces **16 samples for each of the 10 MNIST classes**, with each row corresponding to one digit class.

## Sample Results

The final conditional samples demonstrate class-controlled generation across all ten MNIST digit classes.

![Conditional NCSN Samples](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/05-score-based-models/assets/showcase/conditional_samples.png)

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

* `NCSN_MNIST.ipynb`: The main notebook containing the NCSN and Conditional NCSN implementations, training procedures, sampling methods, and analysis.
* `assets/showcase/`: Selected figures from the notebook used in this README.

## How to Run

1. Install the required packages:

```bash
pip install torch torchvision numpy matplotlib tqdm
```

2. Open `NCSN_MNIST.ipynb` in Jupyter Notebook or Google Colab.
3. Run the notebook cells to download the MNIST dataset, train the models, and generate the visualizations.
4. GPU acceleration is recommended for training and sampling.
