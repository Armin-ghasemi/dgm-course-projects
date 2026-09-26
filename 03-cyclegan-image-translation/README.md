# CycleGAN for Unpaired Image-to-Image Translation

![University](https://img.shields.io/badge/University-University%20of%20Tehran-red)
![Course](https://img.shields.io/badge/Course-Deep%20Generative%20Models-blue)
![Assignment](https://img.shields.io/badge/Assignment-HW1--Q2-green)

This project implements a **CycleGAN** model for unpaired image-to-image translation. Unlike paired image-to-image translation methods, CycleGAN does not require corresponding images between the two domains. Instead, it learns bidirectional mappings and uses **cycle consistency** to preserve the underlying content of the input images.

The main experiment uses the **Horse2Zebra** dataset to learn translations between horse and zebra images. A second experiment on **Monet2Photo** investigates the effect of **identity loss** and a **replay buffer** on the training process.

## Project Goals

1. Implement a **CycleGAN** model for unpaired image-to-image translation.
2. Build bidirectional generators and domain-specific discriminators.
3. Use adversarial and cycle-consistency losses to learn the domain mappings.
4. Train the model on the **Horse2Zebra** dataset.
5. Analyze the training process using loss curves and generated samples.
6. Investigate the effects of **identity loss** and **replay buffers** using the Monet2Photo dataset.

## Task 1: CycleGAN for Unpaired Image-to-Image Translation

### Problem Setup

CycleGAN learns to translate images between two visual domains without requiring paired training examples.

For the main experiment, the two domains are:

* **Domain A:** Horses
* **Domain B:** Zebras

The model learns two mappings:

```text
Horse → Zebra
Zebra → Horse
```

The training data is unpaired, meaning that a horse image and a zebra image provided during the same training step do not need to correspond to each other.

### Data Pipeline

The two domains are loaded independently.

All images are:

* Converted to RGB
* Resized to **128 × 128** pixels
* Randomly horizontally flipped during training
* Normalized using a mean and standard deviation of **0.5** for each RGB channel

The main Horse2Zebra experiment uses a **batch size of 1**.

## CycleGAN Architecture

CycleGAN uses two generators and two discriminators to learn bidirectional translation.

| Component       | Role                   |
| --------------- | ---------------------- |
| Generator A → B | Horse → Zebra          |
| Generator B → A | Zebra → Horse          |
| Discriminator A | Real/Fake Horse images |
| Discriminator B | Real/Fake Zebra images |

### Generator Architecture

Each generator uses a convolutional encoder-decoder architecture with residual blocks.

The architecture contains:

1. An initial convolutional layer with reflection padding
2. Two downsampling layers
3. A sequence of residual blocks
4. Two upsampling layers
5. A final convolutional layer with Tanh activation

The implementation uses **6 residual blocks**.

The residual blocks use skip connections to preserve useful information while learning the domain transformation. **Reflection padding** is used to reduce boundary artifacts, and **Instance Normalization** is applied within the generator.

### Discriminator Architecture

Each domain has its own discriminator.

The implementation uses a **70 × 70 PatchGAN** discriminator. Instead of assigning a single real/fake label to the entire image, the discriminator evaluates local image patches.

This allows the discriminator to focus on local structures and textures that are important for realistic image translation.

## Training Objective

CycleGAN combines adversarial learning with cycle consistency.

### Adversarial Loss

The adversarial loss encourages each generator to produce images that the corresponding discriminator classifies as belonging to the target domain.

The implementation uses **Mean Squared Error (MSE)** for the adversarial objective.

### Cycle-Consistency Loss

Cycle consistency requires an image translated to the other domain and then translated back to remain close to the original image.

For example:

```text
Horse → Zebra → Horse
Zebra → Horse → Zebra
```

The cycle-consistency loss is computed using the **L1 distance** between the original and reconstructed images.

For the main Horse2Zebra experiment, the cycle-consistency term uses:

```text
λ_cycle = 10
```

Identity loss is not used in the main experiment and is investigated separately in the Monet2Photo experiment.

## Main Horse2Zebra Training

The main CycleGAN model is trained on the **Horse2Zebra** dataset.

For each batch:

1. The two generators produce translated images.
2. Adversarial losses are computed for both generators.
3. The generated images are translated back to their original domains.
4. Cycle-consistency loss is computed.
5. The two generators are updated.
6. The two discriminators are updated using real and generated images.

### Training Configuration

| Parameter                 |            Value |
| ------------------------- | ---------------: |
| Dataset                   |      Horse2Zebra |
| Image size                |        128 × 128 |
| Batch size                |                1 |
| Epochs                    |               30 |
| Learning rate             |           0.0002 |
| Optimizer                 |             Adam |
| β1                        |              0.5 |
| β2                        |            0.999 |
| Cycle-consistency weight  |               10 |
| Generator residual blocks |                6 |
| Discriminator             | 70 × 70 PatchGAN |

## Training Analysis

During training, the adversarial, cycle-consistency, discriminator, and total generator losses are recorded for each epoch.

Since CycleGAN is trained through an adversarial optimization process, these losses are not expected to decrease monotonically. The loss curves are therefore analyzed together with the generated images and cycle reconstructions.

### Training Losses

![Horse2Zebra training losses](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/03-cyclegan-image-translation/assets/showcase/horse2zebra_training_losses.png)

### Generated Results

Generated samples are visualized at epochs **10, 20, and 30** in both translation directions:

```text
Horse → Zebra → Horse
Zebra → Horse → Zebra
```

The results provide a qualitative view of the learned domain translation and the preservation of the original image structure.

![Horse2Zebra results](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/03-cyclegan-image-translation/assets/showcase/horse2zebra_results_epoch30.png)

## Task 2: Monet2Photo — Identity Loss and Replay Buffer

A second experiment is performed using the **Monet2Photo** dataset to investigate additional techniques for CycleGAN training.

The two domains are:

* **Domain A:** Monet paintings
* **Domain B:** Real photographs

As in the first experiment, the two domains are unpaired.

The images are resized to **128 × 128** pixels and use the same preprocessing pipeline.

### Replay Buffer

A **replay buffer** is introduced to stabilize discriminator training.

Instead of using only the most recently generated fake images, previously generated samples are stored and occasionally reused when updating the discriminators.

Two separate buffers are used:

* One for generated Monet images
* One for generated photo images

Each buffer has a maximum capacity of **50 images**.

### Identity Loss

Identity loss encourages each generator to preserve an image when the image is already in the target domain.

In this experiment:

```text
λ_identity = 5
```

The identity loss encourages the generators to avoid unnecessary changes to images that are already compatible with the target domain and can help preserve visual characteristics such as color composition.

### Optimized Training

Compared with the main Horse2Zebra experiment, the Monet2Photo training loop adds:

* Identity loss
* Replay buffers
* Mixed-precision training

The generator objective therefore contains adversarial, cycle-consistency, and identity-loss components.

The cycle-consistency term uses:

```text
λ_cycle = 10
```

while the identity term uses:

```text
λ_identity = 5
```

### Monet2Photo Training Configuration

| Parameter                |       Value |
| ------------------------ | ----------: |
| Dataset                  | Monet2Photo |
| Image size               |   128 × 128 |
| Batch size               |           4 |
| Epochs                   |          20 |
| Learning rate            |      0.0002 |
| Cycle-consistency weight |          10 |
| Identity-loss weight     |           5 |
| Replay buffer size       |   50 images |
| Mixed precision          |     Enabled |

### Monet2Photo Results

Generated samples are visualized at epochs **5, 10, and 20** in both directions:

```text
Monet → Photo → Monet
Photo → Monet → Photo
```

These results are used to qualitatively examine the learned translations and cycle reconstructions when identity loss and replay buffers are included.

![Monet2Photo results](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/03-cyclegan-image-translation/assets/showcase/monet2photo_results_epoch20.png)

### Monet2Photo Training Losses

![Monet2Photo training losses](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/03-cyclegan-image-translation/assets/showcase/monet2photo_training_losses.png)

## File Structure

```text
.
├── README.md
├── cyclegan.ipynb
└── assets/
    └── showcase/
        ├── horse2zebra_results_epoch30.png
        ├── horse2zebra_training_losses.png
        ├── monet2photo_results_epoch20.png
        └── monet2photo_training_losses.png
```

## How to Run

1. Clone the repository.
2. Install the required Python dependencies.
3. Open `cyclegan.ipynb` using Jupyter Notebook, JupyterLab, or Google Colab.
4. Run the notebook cells sequentially.
5. The notebook downloads and prepares the required datasets.
6. Run the Horse2Zebra experiment to reproduce the main training results.
7. Run the Monet2Photo experiment to reproduce the identity-loss and replay-buffer analysis.
