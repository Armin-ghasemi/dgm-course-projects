# Masked Autoregressive Flow for Anomaly Detection

![University](https://img.shields.io/badge/University-University%20of%20Tehran-red)
![Course](https://img.shields.io/badge/Course-Deep%20Generative%20Models-blue)
![Assignment](https://img.shields.io/badge/Assignment-HW2--Q1-green)

This project implements a **Masked Autoregressive Flow (MAF)** model for image-based anomaly detection. The model is trained only on normal images from the **MVTec AD** dataset and learns their probability distribution. During inference, the likelihood of each test image is converted into a **Negative Log-Likelihood (NLL)** anomaly score.

The experiment is performed on the **capsule** category of MVTec AD. Normal images are treated as the reference distribution, while defective images are used only during evaluation.

## Project Goals

1. Implement a **Masked Autoregressive Flow** for image likelihood estimation.
2. Build the autoregressive structure using **Masked Autoencoder for Distribution Estimation (MADE)**.
3. Train the flow model only on normal images from the MVTec AD capsule category.
4. Use **Negative Log-Likelihood (NLL)** as an anomaly score.
5. Evaluate anomaly detection performance using **AUROC** and **AUPR**.
6. Analyze the training process, generated samples, and anomaly-score distributions.

## Task 1: Masked Autoregressive Flow for Anomaly Detection

### Problem Setup

The goal of anomaly detection is to identify samples that differ from the normal data distribution.

For this experiment, the model is trained using only normal (`good`) images from the MVTec AD dataset. During testing, the dataset contains both normal images and images with different types of defects.

Instead of learning a classifier for individual defect types, the model learns the probability distribution of normal images. The likelihood of a test image is then used to determine how consistent that image is with the learned normal distribution.

This allows the model to detect defects without using defective images during training.

### Normalizing Flows

A normalizing flow learns an invertible transformation between the data space and a simple latent distribution.

The transformation allows the probability density of an input image to be computed using the change-of-variables formula. The resulting log-likelihood is then converted into an anomaly score:

```text
Anomaly Score = Negative Log-Likelihood
              = -log p(x)
```

Therefore:

* **Low NLL:** the image is more consistent with the learned normal distribution.
* **High NLL:** the image is less likely under the learned normal distribution and is considered more anomalous.

### Masked Autoencoder for Distribution Estimation

The autoregressive structure of the flow is implemented using **MADE**.

In an autoregressive model, the parameters used for each input dimension depend only on preceding dimensions. This dependency is enforced using **masked linear layers**.

The masks prevent information from future dimensions from reaching the current output while preserving the required autoregressive structure.

The implementation uses a custom `MaskedLinear` layer to enforce these connectivity constraints.

### Masked Autoregressive Flow

MAF combines multiple autoregressive transformations to construct a flexible normalizing flow.

For each input dimension, the transformation uses parameters predicted from preceding dimensions. Because the transformation is autoregressive, the log-determinant of its Jacobian can be computed efficiently.

The final model consists of multiple MAF blocks. The ordering of input dimensions is changed between successive blocks so that different dependencies can be learned.

The implemented model uses:

* **7 MAF blocks**
* **512-dimensional hidden layers**
* **128 × 128 RGB input images**
* **Batch Normalization** between flow blocks

## Models & Analysis

A 128 × 128 RGB image contains 49,152 input dimensions. Since the MAF operates directly on the flattened image representation, this results in a very large model with approximately **531.7 million parameters**.

Several techniques are used to make training practical:

* Masked weights are initialized with the corresponding masks applied.
* Gradient hooks keep masked connections inactive during backpropagation.
* Masks are re-applied after optimizer updates.
* Mixed-precision training is used to reduce memory consumption.
* Gradient clipping is applied with a maximum norm of 10.
* The best model weights are kept on the CPU to reduce GPU memory usage.

### Training Configuration

| Parameter              |             Value |
| ---------------------- | ----------------: |
| Dataset                |          MVTec AD |
| Category               |           Capsule |
| Input size             |     128 × 128 × 3 |
| Hidden dimension       |               512 |
| MAF blocks             |                 7 |
| Epochs                 |               100 |
| Batch size             |                 8 |
| Learning rate          |            0.0001 |
| Optimizer              |              Adam |
| LR scheduler           | ReduceLROnPlateau |
| LR reduction factor    |               0.5 |
| Scheduler patience     |          5 epochs |
| Gradient clipping      |                10 |
| Train/Validation split |         80% / 20% |

The training data is divided into training and validation subsets. The learning rate is reduced when the validation loss stops improving.

## Training Objective

The MAF is trained by maximizing the likelihood of the normal training images.

The model transforms an input image into a latent representation and computes its log-likelihood under the learned distribution. The training objective is therefore based on the **Negative Log-Likelihood**.

The training and validation NLL values are tracked throughout training. The validation loss is also used to select the best model checkpoint.

## Sample Results

### Training Analysis

The training and validation NLL curves show how the model learns the distribution of normal images over time. The learning-rate curve shows the effect of the `ReduceLROnPlateau` scheduler.

![Training and validation analysis](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/02-normalizing-flow-anomaly-detection/assets/showcase/training_analysis.png)

The validation loss initially changes substantially before becoming more stable as training progresses. The learning rate is repeatedly reduced when the validation loss stops improving.

### Sampling from the Flow

Because a normalizing flow is invertible, it can also be used in the reverse direction to generate samples.

A latent vector is sampled from a standard Gaussian distribution and passed through the inverse flow transformation to generate an image.

![Samples generated from the MAF](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/02-normalizing-flow-anomaly-detection/assets/showcase/flow_samples.png)

For this experiment, sampling requires sequential computation across the 49,152 input dimensions and is therefore considerably more expensive than likelihood evaluation.

## Anomaly Detection

After training, the flow model is used to compute the likelihood of every test image.

The test set contains:

* `good` images, treated as normal samples
* Images from defect categories, treated as anomalous samples

For each test image, the model computes its likelihood and converts it into an anomaly score using its Negative Log-Likelihood.

The model is **not trained directly on defective test images**. Defect labels are used only during evaluation to measure how well the likelihood-based anomaly score separates normal and anomalous samples.

### NLL Distribution and ROC Curve

![Anomaly detection results](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/02-normalizing-flow-anomaly-detection/assets/showcase/anomaly_detection_results.png)

The NLL distribution shows that many normal images receive relatively concentrated scores, while a portion of anomalous samples receives higher NLL values.

At the same time, there is noticeable overlap between the normal and anomalous score distributions. This indicates that the learned likelihood provides useful information for anomaly detection but does not completely separate all defective samples from the normal distribution.

## Evaluation Metrics

Because anomaly detection datasets are typically imbalanced, accuracy can be misleading. The model is therefore evaluated using **AUROC** and **AUPR**.

### AUROC

The **Area Under the Receiver Operating Characteristic curve** measures how well the anomaly score separates normal and anomalous samples across different decision thresholds.

### AUPR

The **Area Under the Precision-Recall curve** focuses on the precision and recall of anomaly detection and is particularly informative when the classes are imbalanced.

For this experiment, `good` images are assigned the normal label, while all defect categories are treated as anomalies.

### Results

| Metric |      Score |
| ------ | ---------: |
| AUROC  | **75.23%** |
| AUPR   | **92.99%** |

The ROC curve and NLL score distributions provide complementary views of the model's anomaly detection behavior.

## File Structure

```text
.
├── README.md
├── maf_anomaly_detection.ipynb
└── assets/
    └── showcase/
        ├── training_analysis.png
        ├── flow_samples.png
        └── anomaly_detection_results.png
```

## How to Run

1. Clone the repository.
2. Install the required Python dependencies.
3. Download and prepare the **MVTec AD** dataset.
4. Place the dataset in the path expected by the notebook:
   `./mvtec_ad`
5. Make sure the `capsule` category is available under the dataset directory.
6. Open `maf_anomaly_detection.ipynb` using Jupyter Notebook, JupyterLab, or Google Colab.
7. Run the notebook cells sequentially to train the MAF model and reproduce the reported results and visualizations.
