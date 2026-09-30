# Financial Time-Series Generation with Flow Matching

![Course](https://img.shields.io/badge/Course-Deep%20Generative%20Models-blue)
![University](https://img.shields.io/badge/University-University%20of%20Tehran-red)
![Assignment](https://img.shields.io/badge/Assignment-HW4%20--%20Question%202-green)

This project explores generative modeling for financial time series using historical SPY price data and Flow Matching.

Instead of modeling raw prices directly, the data is transformed into log returns and divided into fixed-length windows. A lightweight one-dimensional U-Net is then trained to learn a time-dependent vector field that transforms Gaussian noise toward the financial data distribution.

The generated sequences are evaluated using qualitative comparisons and statistical and temporal properties such as mean, standard deviation, skewness, kurtosis, volatility, Sliced Wasserstein Distance, and autocorrelation.

## Project Goals

* Generate synthetic financial time-series data using Flow Matching.
* Transform historical SPY prices into standardized log-return sequences.
* Train a lightweight one-dimensional U-Net to learn the Flow Matching vector field.
* Increase training-data diversity using offline sequence augmentation.
* Generate synthetic sequences by solving the learned ODE using Euler integration.
* Compare real and generated financial sequences visually.
* Evaluate the generated data using statistical, volatility, distributional, and temporal metrics.

## Task 2: Financial Time-Series Generation with Flow Matching

### Environment and Data Source

The project uses historical SPY price data downloaded through `yfinance`.

The main Python libraries used in the notebook are:

* PyTorch
* NumPy
* Pandas
* Matplotlib
* SciPy
* Seaborn
* yfinance

The model is implemented directly in PyTorch using a lightweight one-dimensional U-Net architecture.

## Data and Preprocessing

Historical SPY data is downloaded over the period from `2010-01-01` to `2023-12-31`.

The adjusted closing price is used as the underlying price series.

The data is divided chronologically into:

* Training set: `90%`
* Test set: `10%`

The dataset contains:

* Total trading days: `3522`
* Training days: `3169`
* Test days: `353`

No missing values or invalid price rows were removed during the cleaning stage.

### Price History

The historical SPY price series and chronological train/test split are visualized below.

![SPY Price History](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/08-flow-matching-time-series/assets/showcase/spy_price_history.png)

## Return Transformation and Windowing

Financial prices are converted into logarithmic returns:

`r_t = log(P_t / P_{t-1})`

Log returns are used instead of raw prices because they describe relative changes and provide a more suitable representation for generative modeling.

The return sequence is standardized using statistics calculated only from the training data:

* Training mean: `0.000485`
* Training standard deviation: `0.010975`

The standardized sequence is then divided into overlapping windows of `64` consecutive observations.

The resulting datasets contain:

* Training windows: `(3104, 64)`
* Test windows: `(289, 64)`

Each training example therefore represents a 64-step financial return sequence.

### Preprocessing Visualization

A representative window is visualized in several forms:

* Real price path
* Simple returns
* Log returns
* Standardized log returns used as the model input

![Preprocessing](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/08-flow-matching-time-series/assets/showcase/preprocessing.png)

## Training Data Augmentation

Additional training samples are generated using offline data augmentation.

The augmentation pipeline combines several transformations:

* Vertical flipping of the return sequence
* Smooth magnitude warping
* Random scaling
* Volatility-adaptive Gaussian noise

Magnitude warping uses a smooth cubic-spline curve, while the added noise is scaled according to the local volatility of each sequence.

Only the training data is augmented. The test data remains unchanged.

The training dataset is expanded by a factor of `4`:

| Dataset | Shape |
| ------- | ----- |
| Original Training Set | `(3104, 64)` |
| Augmented Training Set | `(12416, 64)` |
| Test Set | `(289, 64)` |

### Augmentation Visualization

The original sequence and three augmented versions are shown below.

![Training Data Augmentation](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/08-flow-matching-time-series/assets/showcase/augmentation.png)

## FinanceUNet1D

The generative model is a lightweight one-dimensional U-Net designed specifically for sequential data.

Instead of using 2D convolutions as in image generation, the network uses `Conv1d` layers to process temporal sequences.

The architecture consists of:

* Initial convolution
* Three encoder stages
* Residual convolutional blocks
* Three downsampling operations
* A bottleneck block
* Three transposed-convolution upsampling stages
* Skip connections
* Final `1 × 1` convolution

The model receives:

* A noisy 64-step sequence
* The current flow time `t`

The time value is expanded across the sequence and concatenated with the data as an additional input channel.

The model therefore receives two channels:

* Financial sequence
* Time channel

### Model Configuration

| Parameter | Value |
| --------- | ----- |
| Input Sequence Length | `64` |
| Input Channels | `2` |
| Base Channels | `16` |
| Dropout | `0.3` |
| Convolution Type | Conv1d |
| Normalization | GroupNorm |
| Activation | GELU |
| Residual Blocks | Enabled |
| Skip Connections | Enabled |
| Trainable Parameters | `103,025` |

A dummy forward pass confirms that the model preserves the input sequence shape:

`(8, 64) → (8, 64)`

## Flow Matching Training

During training, each real sequence `x_1` is paired with a random Gaussian noise sample `x_0`.

A random time value `t ∈ [0,1]` is sampled and an interpolated state is constructed:

`x_t = (1 - t)x_0 + tx_1`

The target velocity is:

`v_target = x_1 - x_0`

The U-Net learns to predict this velocity field from the intermediate state and its corresponding time.

The training objective is Mean Squared Error between the predicted and target velocities:

`L = ||v_theta(x_t, t) - (x_1 - x_0)||²`

## Model Training

The model is trained using mini-batch optimization with Adam.

Both training and validation losses are monitored during training.

A `ReduceLROnPlateau` scheduler reduces the learning rate when the validation loss stops improving, while early stopping terminates training after a sufficient number of epochs without improvement.

### Training Configuration

| Parameter | Value |
| --------- | ----- |
| Training Samples | `12,416` |
| Validation/Test Samples | `289` |
| Sequence Length | `64` |
| Batch Size | `64` |
| Maximum Epochs | `150` |
| Initial Learning Rate | `1e-3` |
| Optimizer | Adam |
| Weight Decay | `1e-4` |
| Loss | Mean Squared Error |
| LR Scheduler | ReduceLROnPlateau |
| LR Reduction Factor | `0.5` |
| LR Scheduler Patience | `5` |
| Early Stopping Patience | `20` |
| Gradient Clipping | `1.0` |

The best checkpoint is saved as:

`best_unet_light.pth`

### Training Progress

Training stopped early after epoch `53`.

The best validation loss was:

`1.39665`

The best checkpoint was obtained at epoch `33`.

![Training Progress](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/08-flow-matching-time-series/assets/showcase/training_loss.png)

## ODE-Based Sampling

After training, synthetic financial sequences are generated by solving the learned ordinary differential equation:

`dx/dt = v(x,t)`

The sampling process starts from standard Gaussian noise at `t = 0` and progressively transforms the sequence toward the learned data distribution.

The Euler method is used for numerical integration:

`x_(t+Δt) = x_t + Δt · v(x_t,t)`

The sampling process uses:

* Initial state: Standard Gaussian noise
* Final time: `t = 1`
* Integration method: Euler
* Sampling steps: `100` for the qualitative reconstruction examples
* Step size: `Δt = 1 / steps`

### Sampling and Reconstruction

The generated standardized log returns are progressively converted back into:

1. Standardized log returns
2. Real-scale log returns
3. Reconstructed price paths

![Sampling Reconstruction](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/08-flow-matching-time-series/assets/showcase/sampling_reconstruction.png)

## Real and Generated Sequences

The generated sequences are compared with randomly selected real test windows.

For visualization, both sequences are converted back to price paths using:

`Price_t = P_0 × exp(cumsum(log_returns))`

with an initial price of `100`.

The comparison provides a qualitative check of whether the generated sequences reproduce realistic temporal fluctuations and market-like behavior.

![Real vs Generated](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/08-flow-matching-time-series/assets/showcase/real_vs_generated.png)

## Statistical Evaluation

The generated data is evaluated against the real data using several statistical properties.

The evaluation uses:

* Mean
* Standard deviation
* Skewness
* Kurtosis

A total of `3393` real windows and the same number of synthetic windows are compared.

### Statistical Results

| Metric | Real Data | AI Generated | Difference |
| ------ | --------- | ------------ | ---------- |
| Mean | `0.000483` | `0.000610` | `0.000127` |
| Std Dev | `0.010903` | `0.008497` | `0.002406` |
| Skewness | `-0.728523` | `-0.041160` | `0.687363` |
| Kurtosis | `12.036161` | `7.902679` | `4.133482` |

### Distribution Comparison

The overall distribution of real and generated log returns is visualized below.

![Distribution Comparison](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/08-flow-matching-time-series/assets/showcase/distribution_comparison.png)

## Volatility Analysis

Volatility is analyzed separately to determine whether the generated sequences reproduce the magnitude and variation of fluctuations observed in the real financial data.

The analysis considers:

* Window-level volatility distribution
* Mean volatility
* Maximum volatility
* Minimum volatility
* Rolling 5-day volatility

### Volatility Results

| Metric | Real Data | AI Generated |
| ------ | --------- | ------------ |
| Mean Volatility | `0.010095` | `0.007900` |
| Max Volatility | `0.016966` | `0.024775` |
| Min Volatility | `0.006363` | `0.003782` |

![Volatility Analysis](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/08-flow-matching-time-series/assets/showcase/volatility_analysis.png)

## Distributional and Temporal Comparison

The final evaluation examines both distributional similarity and temporal dependencies.

### Sliced Wasserstein Distance

Sliced Wasserstein Distance (SWD) is calculated by projecting the high-dimensional sequences onto multiple random one-dimensional directions and measuring the resulting distributional distance.

The implementation uses:

* `1000` random projections
* 64-dimensional sequence representations

The resulting SWD is:

`0.000008`

### Autocorrelation

Autocorrelation is calculated for lags from `1` to `20` days.

The autocorrelation curves of the real and generated sequences are then compared using Mean Squared Error.

The resulting Autocorrelation MSE is:

`0.003460`

![Autocorrelation Analysis](https://raw.githubusercontent.com/Armin-ghasemi/dgm-course-projects/main/08-flow-matching-time-series/assets/showcase/autocorrelation.png)

## Models & Analysis

### Data Representation

The model operates on standardized log-return sequences rather than raw financial prices.

Each input sample contains:

* `64` consecutive log returns
* Standardization based on training-set statistics

### Generative Model

**FinanceUNet1D**

A lightweight one-dimensional U-Net with residual blocks, downsampling, upsampling, and skip connections.

The model contains `103,025` trainable parameters.

### Flow Matching

**Flow Matching with Linear Interpolation**

The training process interpolates between Gaussian noise and real data and trains the network to predict the corresponding velocity field.

### Sampling Method

**Euler ODE Integration**

Synthetic sequences are generated by starting from Gaussian noise and numerically integrating the learned vector field from `t = 0` to `t = 1`.

### Evaluation Metrics

The generated sequences are evaluated using:

* Mean
* Standard deviation
* Skewness
* Kurtosis
* Volatility
* Sliced Wasserstein Distance
* Autocorrelation
* Autocorrelation MSE

## Sample Results

The generated sequences are evaluated from several perspectives:

* Reconstruction from Gaussian noise
* Real versus generated price paths
* Overall return distribution
* Volatility distribution
* Temporal autocorrelation

Together, these evaluations examine whether the model reproduces both the statistical characteristics and temporal structure of the original financial data.

## File Structure

```text
08-flow-matching-time-series/
├── flow_matching_financial.ipynb
├── README.md
└── assets/
    └── showcase/
        ├── spy_price_history.png
        ├── preprocessing.png
        ├── augmentation.png
        ├── training_loss.png
        ├── sampling_reconstruction.png
        ├── real_vs_generated.png
        ├── distribution_comparison.png
        ├── volatility_analysis.png
        └── autocorrelation.png
```

## How to Run

1. Clone the repository.

2. Install the required dependencies:

```bash
pip install yfinance matplotlib pandas numpy torch scipy seaborn
```

3. Open:

```text
flow_matching_financial.ipynb
```

4. Run the notebook.

5. The notebook will automatically download historical SPY data using `yfinance`.

6. The notebook will:

   * Download and clean the SPY price data.
   * Split the data chronologically into training and test sets.
   * Convert prices into log returns.
   * Standardize the return sequence.
   * Create 64-step overlapping windows.
   * Augment the training data.
   * Build the FinanceUNet1D model.
   * Train the Flow Matching model.
   * Generate synthetic sequences using Euler ODE integration.
   * Reconstruct price paths.
   * Compare real and generated sequences.
   * Evaluate statistical properties.
   * Analyze volatility.
   * Calculate Sliced Wasserstein Distance.
   * Compare temporal autocorrelation.

7. Save the generated visualization images in:

```text
assets/showcase/
```

## Results

This project demonstrates a Flow Matching approach for generating synthetic financial time-series data from historical SPY price information.

The model operates on standardized 64-step log-return sequences and uses a lightweight one-dimensional U-Net with approximately `103K` trainable parameters.

The trained model generates synthetic sequences by solving a learned ODE from Gaussian noise toward the financial data distribution.

The generated data is evaluated through statistical properties, volatility characteristics, distributional similarity, and temporal autocorrelation. The final evaluation reports an SWD of `0.000008` and an Autocorrelation MSE of `0.003460`.
