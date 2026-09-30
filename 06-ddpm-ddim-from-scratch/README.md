# DDPM and DDIM Diffusion Models on FashionMNIST

![Course](https://img.shields.io/badge/Course-Deep%20Generative%20Models-blue)
![University](https://img.shields.io/badge/University-University%20of%20Tehran-red)
![Assignment](https://img.shields.io/badge/Assignment-HW4%20--%20Question%201%20%28subdivision%201%29-green)

This project implements Denoising Diffusion Probabilistic Models (DDPM) and Denoising Diffusion Implicit Models (DDIM) from scratch for image generation on the FashionMNIST dataset.

The project includes the forward diffusion process, a U-Net based denoising model, DDPM reverse sampling, DDIM sampling, model training, qualitative comparison of generated samples, sampling-speed comparison, and FID-based evaluation.

## Project Goals

* Implement the forward diffusion process from scratch.
* Build a U-Net denoising network for diffusion-based image generation.
* Train a DDPM model on the FashionMNIST dataset.
* Implement the DDPM reverse sampling process.
* Implement DDIM sampling with a reduced number of inference steps.
* Compare DDPM and DDIM in terms of generation quality and sampling speed.
* Evaluate generated samples using the Fréchet Inception Distance (FID).

## Task 1 - Part 1: DDPM and DDIM Diffusion Models

### Environment and Diffusion Setup

The project implements the main components of diffusion models directly in PyTorch.

The implementation includes:

* Forward diffusion
* Noise prediction
* U-Net denoising model
* DDPM reverse sampling
* DDIM sampling
* Model training
* FID evaluation

The diffusion process uses a linear noise schedule with `1000` diffusion timesteps.

The beta schedule is defined by:

* Initial beta: `1e-4`
* Final beta: `0.02`

The model is trained to predict the noise added to the original image at each diffusion timestep.

## Dataset and Data Preparation

The project uses the FashionMNIST dataset.

The images are:

* Resized from `28 x 28` to `32 x 32`.
* Converted to tensors.
* Normalized to the range `[-1, 1]`.

The dataset is downloaded automatically through the PyTorch dataset utilities and does not require manually adding input images to the repository.

The dataset is divided into training and validation sets and loaded using PyTorch DataLoaders.

### Input Images

No manual input images are required for this project.

FashionMNIST is automatically downloaded when the notebook is executed.

### Dataset Preview

A small set of FashionMNIST samples is visualized in the notebook to show the input data distribution.

![FashionMNIST Samples](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/06-ddpm-ddim-from-scratch/assets/showcase/fashionmnist_samples.png)

## U-Net Denoising Model

The denoising network is a U-Net architecture adapted for diffusion models.

The network receives:

* A noisy image `x_t`
* The current diffusion timestep `t`
* The class context information

The architecture uses convolutional blocks, residual connections, timestep embeddings, and context embeddings to predict the noise present in the input image.

### U-Net Architecture

The implemented U-Net contains downsampling and upsampling paths connected through skip connections.

![U-Net Architecture](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/06-ddpm-ddim-from-scratch/assets/architecture/unet_architecture.png)

## Forward Diffusion

The forward diffusion process gradually adds Gaussian noise to the original image over the diffusion timesteps.

For a given timestep `t`, the noisy image is generated using the cumulative product of the diffusion coefficients:

`x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon`

where:

* `x_0` is the original image.
* `x_t` is the noisy image at timestep `t`.
* `alpha_bar_t` is the cumulative product of the alpha values.
* `epsilon` is Gaussian noise.

The model is trained to predict `epsilon` from the noisy image and the corresponding timestep.

## DDPM Sampling

After training, the model generates images by starting from Gaussian noise and progressively removing the predicted noise.

The DDPM reverse process uses all `1000` diffusion timesteps to transform a random noise image into a generated FashionMNIST sample.

The reverse sampling process follows the standard DDPM formulation and uses the trained U-Net to predict the noise at each timestep.

## Model Training

The U-Net is trained using Mean Squared Error (MSE) between the true noise and the noise predicted by the network.

The optimizer is Adam and the learning rate is adjusted using a cosine annealing scheduler.

### Training Configuration

| Parameter | Value |
| --------- | ----- |
| Dataset | FashionMNIST |
| Image Size | `32 x 32` |
| Channels | `1` |
| Diffusion Steps | `1000` |
| Beta Start | `1e-4` |
| Beta End | `0.02` |
| Feature Dimension | `64` |
| Context Dimension | `10` |
| Batch Size | `128` |
| Epochs | `40` |
| Learning Rate | `5e-4` |
| Optimizer | Adam |
| LR Scheduler | Cosine Annealing |
| Loss | Mean Squared Error |

### Training Loss

The training loss decreases throughout training, reaching its minimum value at approximately epoch `37`.

![Training Loss](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/06-ddpm-ddim-from-scratch/assets/showcase/training_loss.png)

The minimum recorded training loss is approximately:

`0.02407`

## DDIM Sampling

DDIM provides an alternative sampling procedure that can generate images using substantially fewer sampling steps than DDPM.

In this project, DDIM sampling is performed using `100` inference steps instead of the `1000` steps used by DDPM.

This allows the generation process to be significantly faster while still producing meaningful FashionMNIST samples.

### DDIM Sampling Algorithm

The DDIM sampling procedure implemented in the project follows the deterministic reverse diffusion formulation.

![DDIM Sampling Algorithm](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/06-ddpm-ddim-from-scratch/assets/architecture/ddim_algorithm.png)

## Results and Evaluation

The trained model is evaluated using both DDPM and DDIM sampling.

The generated samples are compared using:

* Qualitative visual inspection
* Sampling time
* FID score

Two model checkpoints are considered:

* Best Model: checkpoint corresponding to the lowest training loss.
* Last Model: checkpoint from the final training epoch.

## Models & Analysis

### Best Model + DDPM

The best-performing training checkpoint is sampled using the standard DDPM reverse process with `1000` diffusion steps.

![Best Model DDPM](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/06-ddpm-ddim-from-scratch/assets/showcase/best_ddpm.png)

### Best Model + DDIM

The same best checkpoint is sampled using DDIM with only `100` inference steps.

![Best Model DDIM](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/06-ddpm-ddim-from-scratch/assets/showcase/best_ddim.png)

### Last Model + DDPM

The final training checkpoint is sampled using DDPM with `1000` diffusion steps.

![Last Model DDPM](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/06-ddpm-ddim-from-scratch/assets/showcase/last_ddpm.png)

### Last Model + DDIM

The final training checkpoint is sampled using DDIM with `100` inference steps.

![Last Model DDIM](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/06-ddpm-ddim-from-scratch/assets/showcase/last_ddim.png)

### Sampling Speed

The sampling procedures are compared using the same number of generated samples.

| Sampling Method | Steps | Time |
| --------------- | ----- | ---- |
| DDPM | `1000` | `20.6 s` |
| DDIM | `100` | `2.03 s` |

The DDIM implementation requires substantially less sampling time because it uses only `100` inference steps instead of the `1000` steps used by DDPM.

## FID Evaluation

The generated samples are also evaluated using the Fréchet Inception Distance (FID).

For the evaluation:

* `200` real FashionMNIST samples are used.
* `200` generated samples are used.
* Inception features are extracted with feature dimension `2048`.

### FID Results

| Model | Sampling Method | FID |
| ----- | --------------- | --- |
| Best Model | DDPM | `169.8301` |
| Best Model | DDIM | `313.3063` |
| Last Model | DDPM | `146.7604` |
| Last Model | DDIM | `179.2615` |

The FID values provide a quantitative comparison between the different checkpoints and sampling procedures used in the project.

## Inference Parameters

The final qualitative comparison uses the following sampling configurations:

* DDPM: `1000` sampling steps.
* DDIM: `100` sampling steps.
* Best Model: checkpoint with the minimum training loss.
* Last Model: checkpoint saved at the end of training.

These configurations allow both the trained model checkpoint and the sampling algorithm to be compared.

## Sample Results

The generated samples demonstrate that the trained diffusion model can synthesize FashionMNIST-like images from random noise.

The results include comparisons between:

* Best Model + DDPM
* Best Model + DDIM
* Last Model + DDPM
* Last Model + DDIM

The comparison also demonstrates the trade-off between the number of sampling steps and generation time.

## File Structure

```text
06-ddpm-ddim-from-scratch/
├── ddpm_ddim.ipynb
├── README.md
└── assets/
    ├── architecture/
    │   ├── unet_architecture.png
    │   └── ddim_algorithm.png
    └── showcase/
        ├── fashionmnist_samples.png
        ├── training_loss.png
        ├── best_ddpm.png
        ├── best_ddim.png
        ├── last_ddpm.png
        └── last_ddim.png
```

## How to Run

1. Clone the repository.

2. Install the required dependencies:

```bash
pip install torch torchvision numpy matplotlib scipy scikit-learn tqdm
```

3. Open:

```text
ddpm_ddim.ipynb
```

4. Run the notebook.

5. The FashionMNIST dataset will be downloaded automatically.

6. The notebook will:

   * Prepare the FashionMNIST dataset.
   * Visualize sample input images.
   * Define the diffusion noise schedule.
   * Build the U-Net denoising model.
   * Train the diffusion model.
   * Track the training loss.
   * Generate samples using DDPM.
   * Generate samples using DDIM.
   * Compare sampling speed.
   * Evaluate generated samples using FID.

7. Save the generated visualization images in:

```text
assets/showcase/
```

8. Save the architecture-related figures in:

```text
assets/architecture/
```

## Results

This project demonstrates a from-scratch implementation of diffusion-based image generation using DDPM and DDIM on FashionMNIST.

The trained U-Net successfully learns to predict the noise added during the forward diffusion process and can generate new samples through the reverse diffusion process.

The experiments also demonstrate the difference between DDPM and DDIM sampling. DDPM uses `1000` reverse diffusion steps, while DDIM generates samples using only `100` steps, resulting in a substantially shorter sampling time in this experiment.

The project further compares the best and last training checkpoints using both sampling methods and evaluates the generated samples quantitatively using FID.
