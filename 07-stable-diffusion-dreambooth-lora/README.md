# DreamBooth Fine-tuning with LoRA

![Course](https://img.shields.io/badge/Course-Deep%20Generative%20Models-blue)
![University](https://img.shields.io/badge/University-University%20of%20Tehran-red)
![Assignment](https://img.shields.io/badge/Assignment-HW4%20--%20Question%201%20%28Part%202%29-green)

This project focuses on fine-tuning a pretrained Stable Diffusion v1.5 model for subject-specific image generation using DreamBooth and Low-Rank Adaptation (LoRA).

The model is fine-tuned using five reference images of a specific toy. DreamBooth is used for subject personalization and prior preservation, while LoRA provides a parameter-efficient way to adapt the pretrained diffusion model.

## Project Goals

* Fine-tune Stable Diffusion v1.5 for a specific subject using DreamBooth.
* Use LoRA for parameter-efficient fine-tuning.
* Preserve the general concept of the subject class using prior preservation.
* Generate the learned subject under different prompts and styles.
* Analyze the effect of classifier-free guidance scale and inference steps on generated images.

## Task 1 - Part 2: DreamBooth Fine-tuning with LoRA

### Environment and Pretrained Model

The project uses the Hugging Face Diffusers ecosystem for diffusion-model fine-tuning.

The main pretrained model is:

* Stable Diffusion v1.5

The implementation uses:

* Diffusers
* Transformers
* Accelerate
* PEFT
* BitsAndBytes
* xFormers

The pretrained Stable Diffusion model is not implemented from scratch. Instead, the project adapts the existing model to a new subject using DreamBooth and LoRA.

## Subject and Data Preparation

Five reference images are used as the instance data for DreamBooth fine-tuning.

The instance prompt is:

`a photo of a sks toy`

The placeholder token `sks` is used to associate the learned concept with the specific training subject.

The images are prepared at a resolution of `512 x 512` using center cropping and normalization to the range `[-1, 1]`.

### Input Images

The five training images are not embedded in the original notebook HTML. They should therefore be added manually to the project.

Place them in:

```text
assets/input/
```

with the following names:

```text
assets/input/
├── instance_01.png
├── instance_02.png
├── instance_03.png
├── instance_04.png
└── instance_05.png
```

The notebook expects the five images to be available through:

```text
./instance_data
```

The files in `assets/input/` should therefore be copied or linked to the location expected by the notebook before running the training process.

### Training Images

| Image       | Preview                                                                                                                                  |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Instance 01 | ![Instance 01](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/07-dreambooth-lora/assets/input/instance_01.png) |
| Instance 02 | ![Instance 02](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/07-dreambooth-lora/assets/input/instance_02.png) |
| Instance 03 | ![Instance 03](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/07-dreambooth-lora/assets/input/instance_03.png) |
| Instance 04 | ![Instance 04](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/07-dreambooth-lora/assets/input/instance_04.png) |
| Instance 05 | ![Instance 05](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/07-dreambooth-lora/assets/input/instance_05.png) |

## Prior Preservation

Prior preservation is enabled to help maintain the general concept of the subject class while learning the specific instance.

The class prompt is:

`a photo of a toy`

The training process uses:

* 100 class images
* Prior loss weight: `1.0`

The overall objective combines the instance-specific loss with the prior-preservation loss:

`L = L_instance + λ × L_prior`

This encourages the model to learn the specific toy without losing the general concept of the class.

## Dataset and Tokenization

The training process uses two types of data:

1. Instance images representing the specific toy.
2. Class images representing the general toy concept.

The instance prompt uses the placeholder token:

`a photo of a sks toy`

while the class prompt uses:

`a photo of a toy`

The tokenizer converts these prompts into token representations used by the text-conditioning component of Stable Diffusion.

## Stable Diffusion and LoRA

The project starts from the pretrained Stable Diffusion v1.5 model.

Instead of updating the complete diffusion model, LoRA adapters are injected into selected attention-related layers.

The LoRA configuration includes:

* Rank: `4`
* Alpha: `4`
* Initialization: Gaussian
* Target modules:

  * `to_k`
  * `to_q`
  * `to_v`
  * `to_out.0`
  * `add_k_proj`
  * `add_v_proj`

This significantly reduces the number of parameters that need to be updated during fine-tuning.

## Diffusion Training

The diffusion model is trained to predict the noise added to the clean image during the forward diffusion process.

The main noise-prediction objective can be represented as:

`L = ||ε - εθ(zt, t, c)||²`

where:

* `ε` is the sampled noise.
* `εθ` is the model's predicted noise.
* `zt` is the noisy latent representation.
* `t` is the diffusion timestep.
* `c` represents the text conditioning.

With prior preservation enabled, the instance-specific and class-specific objectives are combined during training.

## Fine-tuning Configuration

The main training configuration is:

| Parameter             | Value                 |
| --------------------- | --------------------- |
| Base Model            | Stable Diffusion v1.5 |
| Instance Images       | 5                     |
| Resolution            | 512 × 512             |
| Batch Size            | 1                     |
| Training Steps        | 1000                  |
| Epochs                | 10                    |
| Learning Rate         | 1e-4                  |
| LR Scheduler          | Constant              |
| LoRA Rank             | 4                     |
| LoRA Alpha            | 4                     |
| Prior Preservation    | Enabled               |
| Class Images          | 100                   |
| Prior Loss Weight     | 1.0                   |
| Gradient Accumulation | 1                     |
| Max Gradient Norm     | 1.0                   |
| Mixed Precision       | FP16                  |
| 8-bit Adam            | Enabled               |
| Text Encoder Training | Enabled               |
| Checkpoint Frequency  | Every 250 steps       |

The resulting LoRA weights are saved to:

```text
./lora-dreambooth-model
```

## Inference and Analysis

After fine-tuning, the learned LoRA weights are used to generate images with different prompts.

The inference experiments compare:

* Guidance scales: `5.0`, `7.5`, `10.0`
* Inference steps: `30`, `50`, `100`

Each output image is a `3 × 3` grid where:

* Columns correspond to guidance scales.
* Rows correspond to the number of inference steps.

### Prompt 1 - Base Subject

`a photo of a sks toy`

![Base Subject](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/07-dreambooth-lora/assets/showcase/prompt_base.png)

### Prompt 2 - Class Prompt

`a photo of a toy`

![Class Prompt](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/07-dreambooth-lora/assets/showcase/prompt_class.png)

### Prompt 3 - Reading a Book

`a photo of a sks toy reading a book`

![Reading a Book](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/07-dreambooth-lora/assets/showcase/prompt_reading_book.png)

### Prompt 4 - In a Bucket

`a photo of a sks toy in a bucket`

![Toy in a Bucket](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/07-dreambooth-lora/assets/showcase/prompt_bucket.png)

### Prompt 5 - Artistic Style

`oil painting of a sks toy in starry night style`

![Starry Night Style](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/07-dreambooth-lora/assets/showcase/prompt_starry_night.png)

## Models & Analysis

### Base Model

**Stable Diffusion v1.5**

A pretrained latent diffusion model used as the starting point for subject-specific fine-tuning.

### Fine-tuning Method

**DreamBooth + LoRA**

DreamBooth provides subject personalization, while LoRA enables parameter-efficient adaptation of the pretrained model.

### Prior Preservation

Prior preservation uses class images to maintain the general concept of the subject class during personalization.

### Inference Parameters

The generated results are evaluated across different combinations of:

* Guidance scale: `5.0`, `7.5`, `10.0`
* Inference steps: `30`, `50`, `100`

These experiments provide a visual comparison of how inference settings affect the generated subject and its adherence to the prompt.

## Sample Results

The following examples demonstrate the learned subject under different prompts:

* Base subject generation
* General class generation
* Subject reading a book
* Subject inside a bucket
* Subject rendered in an artistic style

The results show that the fine-tuned model can generate the learned subject in multiple contexts while retaining the main visual characteristics learned from the five reference images.

## File Structure

```text
07-dreambooth-lora/
├── dreambooth_lora.ipynb
├── README.md
└── assets/
    ├── input/
    │   ├── instance_01.png
    │   ├── instance_02.png
    │   ├── instance_03.png
    │   ├── instance_04.png
    │   └── instance_05.png
    └── showcase/
        ├── prompt_base.png
        ├── prompt_class.png
        ├── prompt_reading_book.png
        ├── prompt_bucket.png
        └── prompt_starry_night.png
```

## How to Run

1. Clone the repository.

2. Install the required dependencies:

```bash
pip install diffusers transformers accelerate peft bitsandbytes xformers
```

3. Place the five instance images in:

```text
assets/input/
```

4. Make sure the images are also available at the location expected by the notebook:

```text
./instance_data
```

5. Open:

```text
dreambooth_lora.ipynb
```

6. Run the notebook to:

   * Prepare the instance and class data.
   * Load Stable Diffusion v1.5.
   * Inject LoRA adapters.
   * Fine-tune the model using DreamBooth.
   * Apply prior preservation.
   * Save the trained LoRA weights.
   * Generate images with different prompts, guidance scales, and inference steps.

7. Save the generated showcase images in:

```text
assets/showcase/
```

## Results

This project demonstrates subject-specific fine-tuning of Stable Diffusion v1.5 using DreamBooth and LoRA. Starting from only five reference images, the fine-tuned model can generate the learned subject under different textual prompts and visual contexts.

The inference experiments also provide a comparison of different guidance scales and inference-step configurations for the personalized model.
