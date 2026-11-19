# Comparative Implementation and Empirical Evaluation of Machine Learning Regression Algorithms

**Course:** Machine Learning / Computer Vision (CS / AI Laboratory Series)  
**Author:** Atif Ali Shah  
**Student ID:** FA23-BAI-004  
**Degree Program:** BS Artificial Intelligence  
**Institution:** Department of Computer Science & Artificial Intelligence  
**Repository:** [AtifAliShah/ComputerVision-LABS](https://github.com/AtifAliShah/ComputerVision-LABS)  
**Deliverable Path:** `lab10/Lab10_Report.md`  

---

## Abstract
Regression analysis represents a cornerstone of supervised statistical learning, tasked with inferring a continuous mapping function $f: \mathbb{R}^d \to \mathbb{R}$ between multi-dimensional feature vectors and numerical target outputs. This laboratory investigation provides a rigorous, empirical, and theoretical comparative assessment of six representative regression paradigms: Ordinary Least Squares (OLS) Linear Regression, Decision Tree Regression (CART), Random Forest Regression, Multilayer Perceptron Neural Network Regression (MLPRegressor), Extreme Gradient Boosting (XGBoost), and K-Nearest Neighbors Regression (KNN). All algorithms are systematically trained and cross-evaluated on the standard California Housing dataset (20,640 census records) under an identical 80:20 train-test partition (`random_state=42`) with strict data-leakage controls. Performance is benchmarked across four standard statistical metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and the Coefficient of Determination ($R^2$). Experimental findings establish that **XGBoost achieves state-of-the-art predictive accuracy ($R^2 = 0.8350$, $\text{RMSE} = 0.4650$, $\text{MAE} = 0.3091$)**, substantially outperforming parametric linear models ($R^2 = 0.5758$) and non-parametric instance-based learners ($R^2 = 0.6801$). The architectural characteristics, geometric loss surfaces, regularization dynamics, and residual error distributions governing each model's behavior are analyzed in detail.

---

## 1. Introduction
In real-world data science and computer vision pipelines, continuous estimation tasks—such as bounding box coordinate regression, depth estimation, human pose coordinate modeling, and real estate valuation—demand models that can capture complex, non-linear relationships across heterogeneous feature spaces. While linear models provide high interpretability and analytical tractability, they frequently underfit complex manifold distributions characterized by localized spatial interactions and non-linear feature coupling. Conversely, non-parametric tree ensembles and deep artificial neural networks offer universal function approximation capabilities but exhibit distinct vulnerabilities regarding variance, hyperparameter sensitivity, and computational overhead.

The objective of this research report is to evaluate how distinct inductive biases affect regression performance on a standardized benchmark. By enforcing uniform data splitting, standardized scaling protocols where theoretically warranted, and rigorous residual analysis, this study decouples algorithm capability from preprocessing discrepancies, providing an empirical benchmark of modern regression algorithms.

---

## 2. Objectives
The explicit objectives of this laboratory study are:
1. **Architectural Implementation:** Implement six distinct regression algorithms spanning parametric, non-parametric, ensemble (bagging and boosting), neural, and instance-based machine learning paradigms.
2. **Standardized Benchmarking:** Evaluate all six models under identical experimental conditions using the California Housing dataset with an 80% training and 20% hold-out testing partition.
3. **Leakage-Free Preprocessing:** Construct scikit-learn `Pipeline` architectures ensuring that feature scaling transformations (`StandardScaler`) are fit exclusively on training data folds.
4. **Multi-Metric Quantitative Evaluation:** Measure and contrast model fidelity using Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and the Coefficient of Determination ($R^2$).
5. **Empirical Visualization:** Generate publication-quality visualizations, including target distributions, feature correlation matrices, Actual vs. Predicted scatter plots with diagonal reference lines, residual error diagnostic plots, and comparative metric distributions.
6. **Theoretical & Error Diagnostic Analysis:** Dissect model strengths, structural limitations, computational complexities, and residual patterns to explain the empirical dominance of gradient-boosted decision trees over linear, neural, and distance-based regressors.

---

## 3. Problem Statement
Given a set of $n$ observation tuples $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$, where $\mathbf{x}_i \in \mathbb{R}^d$ denotes a $d$-dimensional vector of socioeconomic, demographic, and geospatial attributes, and $y_i \in \mathbb{R}^+$ denotes the true continuous median house value of the $i$-th district (in units of $\$100,000$), the objective is to learn an optimal predictive mapping:

$$\hat{y}_i = f(\mathbf{x}_i; \mathbf{\Theta})$$

such that the empirical risk over the unobserved test distribution $\mathcal{D}_{\text{test}}$ is minimized with respect to a designated loss function (e.g., squared Euclidean loss $\mathcal{L}(y, \hat{y}) = (y - \hat{y})^2$). The learning system must exhibit minimal bias, low variance, and generalizability without overfitting training noise or geographical outliers.

---

## 4. Dataset Description
The experimental benchmark utilized throughout this investigation is the **California Housing Dataset**, originally constructed by Pace and Barry (1997) from the 1990 U.S. Census and distributed via the `scikit-learn` dataset repository. 

The dataset captures geographic, infrastructural, and economic variables aggregate across 20,640 census block groups (districts) in California. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data, typically encapsulating a population of 600 to 3,000 individuals.

```text
Dataset Overview:
---------------------------------------------
Total Samples (Rows):       20,640
Total Features (Columns):   8 continuous attributes
Target Variable:            MedHouseVal (continuous)
Missing Values (NaN):       0 across all columns
Duplicate Records:          0 records
Data Format:                64-bit IEEE Floating Point (float64)
---------------------------------------------
```

---

## 5. Features and Target Variable
The feature matrix $\mathbf{X} \in \mathbb{R}^{20640 \times 8}$ comprises eight continuous explanatory variables, while the target vector $\mathbf{y} \in \mathbb{R}^{20640}$ represents district house valuation.

| Feature Identifier | Canonical Variable Name | Statistical Type | Measurement Unit | Domain Description |
|:---|:---|:---:|:---:|:---|
| $x_1$ | `MedInc` | Continuous | $\$10,000\text{s}$ | Median household income within the census block group |
| $x_2$ | `HouseAge` | Continuous | Years | Median age of housing structures in the block group |
| $x_3$ | `AveRooms` | Continuous | Ratio (Rooms/HH) | Average count of rooms per residential household |
| $x_4$ | `AveBedrms` | Continuous | Ratio (Bedrooms/HH) | Average count of dedicated bedrooms per residential household |
| $x_5$ | `Population` | Continuous | Persons | Total resident population count inhabiting the block group |
| $x_6$ | `AveOccup` | Continuous | Ratio (Persons/HH) | Average number of household members residing together |
| $x_7$ | `Latitude` | Continuous | Decimal Degrees ($^\circ\text{N}$) | Geographic latitude coordinate of block group centroid |
| $x_8$ | `Longitude` | Continuous | Decimal Degrees ($^\circ\text{W}$) | Geographic longitude coordinate of block group centroid |
| **$y$ (Target)** | **`MedHouseVal`** | **Continuous** | **$\$100,000\text{s}$** | **Median monetary value of owner-occupied houses** |

### Target Variable Characteristics and Distribution Analysis
The target variable `MedHouseVal` ranges from a minimum of $\$0.14999$ ($\$14,999$) to a maximum of $\$5.00001$ ($\$500,001$). A notable characteristic of the California Housing dataset is the artificial upper truncation (right-censoring) at $\$5.00001$, representing houses valued at $\$500,000$ or above that were capped during census collection. The distribution exhibits a parametric mean of $\mu = 2.0686$ ($\approx \$206,856$) and a median of $\text{Med} = 1.7970$ ($\approx \$179,700$), indicating positive (rightward) skewness.

![Target Distribution](results/figures/target_distribution.png)

### Correlation and Multicollinearity Analysis
To evaluate linear associations, Pearson correlation coefficients were computed across all feature pairings and against the target variable `MedHouseVal`.

![Correlation Heatmap](results/figures/correlation_heatmap.png)

Key observations from the correlation matrix include:
1. **Dominant Linear Predictor (`MedInc`):** `MedInc` exhibits the strongest linear correlation with `MedHouseVal` ($r = +0.69$), confirming that district income level is the primary linear driver of real estate valuation.
2. **Geographical Coordinates (`Latitude`, `Longitude`):** `Latitude` and `Longitude` display weak univariate linear correlations with price ($r = -0.14$ and $r = -0.05$ respectively), but exhibit high mutual collinearity ($r = -0.92$). Their relationship with property value is spatial and non-linear: coastal proximity (San Francisco Bay Area and coastal Southern California) commands massive premiums over inland regions at similar latitudes.
3. **Room and Bedroom Collinearity:** `AveRooms` and `AveBedrms` correlate strongly with each other ($r = +0.85$), reflecting architectural scaling within households.

---

## 6. Tools and Technologies Stack
The technical implementation utilizes the Python open-source scientific computing ecosystem:
- **Python (v3.13.9):** High-level execution runtime.
- **NumPy (v2.2.0):** Vectorized mathematical array operations and linear algebra routines.
- **Pandas (v2.2.3):** DataFrame manipulation, descriptive statistical summaries, and tabular serialization.
- **Scikit-Learn (v1.6.0):** Data splitting (`train_test_split`), standardization (`StandardScaler`), pipeline assembly (`Pipeline`), baseline model implementations (`LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`, `MLPRegressor`, `KNeighborsRegressor`), and evaluation metrics (`mean_absolute_error`, `mean_squared_error`, `r2_score`).
- **XGBoost (v2.1.3):** Scalable, distributed, regularized gradient boosted decision tree library.
- **Matplotlib (v3.10.0) & Seaborn (v0.13.2):** High-resolution (300 DPI) visualization generation.
- **Jupyter Notebook (`nbformat`, `nbconvert`):** Interactive, reproducible workflow execution.

---

## 7. Data Preprocessing & Leakage Prevention
Data preprocessing is executed under strict statistical isolation protocols:
1. **Separation of Variables:** Independent predictor matrix $\mathbf{X}$ is partitioned from the target scalar vector $\mathbf{y}$.
2. **Train-Test Partitioning:** An exact 80:20 train-test ratio is established via `train_test_split` with a fixed seed `random_state=42`. This yields:
   - Training partition: $N_{\text{train}} = 16,512$ samples ($80.0\%$)
   - Testing partition: $N_{\text{test}} = 4,128$ samples ($20.0\%$)
3. **Feature Scaling Strategy:**
   Feature scaling is necessary for algorithms whose loss formulations or distance functions depend on feature scale:
   - **Linear Regression:** Requires `StandardScaler` so that gradient updates or normal equation matrix inversions avoid ill-conditioned conditioning numbers across disparate scales (e.g., `Population` $\in [3, 35682]$ vs. `AveBedrms` $\in [0.33, 34.07]$).
   - **Neural Networks (MLPRegressor):** Requires `StandardScaler` to prevent backpropagated gradients from exploding or vanishing across unbounded input dimensions.
   - **K-Nearest Neighbors (KNN):** Requires `StandardScaler` because Euclidean distance $d(\mathbf{u}, \mathbf{v}) = \sqrt{\sum (u_j - v_j)^2}$ is otherwise dominated by large-magnitude attributes like `Population`.
   - **Decision Tree, Random Forest, and XGBoost:** Tree splitting criteria (variance reduction) are invariant to monotonic scaling; unscaled raw features preserve integer thresholds without numerical distortion.
4. **Data Leakage Mitigation:**
   To prevent test set statistics from biasing training normalization, all `StandardScaler` transformations are wrapped in scikit-learn `Pipeline` structures. The mean $\boldsymbol{\mu}_{\text{train}}$ and variance $\boldsymbol{\sigma}_{\text{train}}^2$ are derived strictly from $X_{\text{train}}$, with the identical transformation applied downstream to $X_{\text{test}}$:

$$\mathbf{x}_{\text{norm}} = \frac{\mathbf{x} - \boldsymbol{\mu}_{\text{train}}}{\boldsymbol{\sigma}_{\text{train}}}$$

---

## 8. Experimental Methodology
To guarantee reproducibility across all six models:
1. The identical training fold $\mathbf{X}_{\text{train}}, \mathbf{y}_{\text{train}}$ and hold-out test fold $\mathbf{X}_{\text{test}}, \mathbf{y}_{\text{test}}$ are shared across all six algorithms.
2. Random state seeds are uniformly fixed (`random_state=42`) across stochastic procedures (data split, Decision Tree split randomizer, Random Forest bagging bootstrapping, MLP weight initialization, and XGBoost subsampling).
3. Predictions $\hat{\mathbf{y}}_{\text{test}}$ are obtained strictly by invoking `.predict(X_test)` on the trained estimators.
4. Evaluation metrics (MAE, MSE, RMSE, $R^2$) are computed against the identical ground-truth vector $\mathbf{y}_{\text{test}}$.
5. Execution wall-clock durations for model training are logged to assess computational efficiency.

```mermaid
flowchart TD
    A["California Housing Dataset\n(20,640 records, 8 features)"] --> B["Train-Test Partition (80:20)\nrandom_state=42"]
    B --> C["Training Set (16,512 samples)"]
    B --> D["Hold-out Testing Set (4,128 samples)"]
    
    C --> E1["StandardScaler Pipeline\nFit on Train Only"]
    C --> E2["Raw Feature Pipeline\nTree Invariance"]
    
    E1 --> M1["1. Linear Regression"]
    E2 --> M2["2. Decision Tree"]
    E2 --> M3["3. Random Forest"]
    E1 --> M4["4. Neural Network (MLP)"]
    E2 --> M5["5. XGBoost Regressor"]
    E1 --> M6["6. KNN Regressor"]
    
    M1 --> P["Generate Test Predictions\ny_hat across 6 models"]
    M2 --> P
    M3 --> P
    M4 --> P
    M5 --> P
    M6 --> P
    
    D --> EVAL["Evaluate Performance Metrics:\nMAE, MSE, RMSE, R² Score"]
    P --> EVAL
    EVAL --> VIS["Diagnostic Visualizations &\nResidual Analysis"]
```

---

## 9. Description of All Six Algorithms

### 9.1 Ordinary Least Squares (OLS) Linear Regression
- **Mechanism:** Linear regression models the dependent variable as an affine linear combination of input predictors:
  $$\hat{y}_i = \beta_0 + \sum_{j=1}^d \beta_j x_{ij} = \mathbf{x}_i^T \boldsymbol{\beta}$$
  Parameters $\boldsymbol{\beta} \in \mathbb{R}^{d+1}$ are determined analytically by solving the Normal Equation $\boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$, which minimizes the residual sum of squares (RSS).
- **Suitability for Regression:** Establishes the baseline parametric standard. Provides closed-form optimization and direct parameter interpretability.
- **Advantages:** Computationally instantaneous $\mathcal{O}(nd^2 + d^3)$; zero hyperparameter tuning required; mathematically well-conditioned; easily interpretable regression coefficients.
- **Limitations:** Enforces a rigid hyper-planar assumption. Completely incapable of capturing localized spatial clusters (e.g., coastal micro-climates) or non-linear interaction terms without manual feature expansion.

### 9.2 Decision Tree Regression (CART)
- **Mechanism:** Classification and Regression Trees (CART) partition the feature space into $M$ orthogonal hyper-rectangles $R_1, R_2, \dots, R_M$. At each node, the feature $j$ and split threshold $s$ are chosen to minimize the sum of squared errors across resulting child partitions:
  $$\min_{j, s} \left[ \sum_{\mathbf{x}_i \in R_{\text{left}}(j, s)} (y_i - \bar{y}_{\text{left}})^2 + \sum_{\mathbf{x}_i \in R_{\text{right}}(j, s)} (y_i - \bar{y}_{\text{right}})^2 \right]$$
  The prediction for any sample falling into region $R_m$ is the empirical mean $\hat{c}_m = \frac{1}{N_m} \sum_{\mathbf{x}_i \in R_m} y_i$.
- **Suitability for Regression:** Handles non-linear relationships, multi-modal distributions, and threshold-based decision boundaries without requiring monotonic transformations.
- **Advantages:** Highly interpretable hierarchical rules; scale-invariant; automatically captures complex feature interactions.
- **Limitations:** Prone to high variance and catastrophic overfitting on deep branches; produces piecewise-constant step-function approximations rather than smooth continuous gradients.

### 9.3 Random Forest Regression
- **Mechanism:** Random Forest is an ensemble bagging (Bootstrap Aggregation) technique that constructs an ensemble of $B$ decorrelated decision trees $\{T_b\}_{b=1}^B$. Each tree is trained on an independently drawn bootstrap sample $\mathcal{D}_b^*$ of the training data. At each candidate split, a random subset of $m \approx \sqrt{d}$ or $d/3$ features is evaluated. Final predictions average individual tree outputs:
  $$\hat{y}_{\text{RF}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x})$$
- **Suitability for Regression:** Reduces variance by averaging multiple deep, low-bias decision trees.
- **Advantages:** Substantially reduces prediction variance without inflating bias; robust against outliers; resistant to overfitting; provides reliable feature importance metrics.
- **Limitations:** Increased computational and memory footprint during inference; loss of single-tree interpretability; inability to extrapolate beyond the maximum target values seen during training.

### 9.4 Neural Network Regression (MLPRegressor)
- **Mechanism:** A Multilayer Perceptron is a feedforward artificial neural network consisting of an input layer, one or more hidden layers with non-linear activation functions, and a linear output neuron:
  $$\mathbf{h}^{(1)} = \text{ReLU}(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)})$$
  $$\mathbf{h}^{(2)} = \text{ReLU}(\mathbf{W}^{(2)} \mathbf{h}^{(1)} + \mathbf{b}^{(2)})$$
  $$\hat{y} = \mathbf{w}^{(3)T} \mathbf{h}^{(2)} + b^{(3)}$$
  Optimization is conducted via Adam (Adaptive Moment Estimation), backpropagating mean squared error gradients across weight tensors using second-order moment estimation.
- **Suitability for Regression:** Acts as a universal function approximator capable of synthesizing continuous, smooth high-dimensional manifolds.
- **Advantages:** Capable of modeling complex joint continuous representations; scales well with large datasets; smooth differentiable output surface.
- **Limitations:** Computationally intensive to train; sensitive to weight initialization and learning rates; susceptible to local minima; strictly demands normalized/standardized inputs.

### 9.5 Extreme Gradient Boosting Regression (XGBoost)
- **Mechanism:** XGBoost is a sequential boosting framework that builds an additive expansion of regression trees $\hat{y}_i^{(K)} = \sum_{k=1}^K f_k(\mathbf{x}_i)$. At step $t$, the algorithm minimizes a regularized objective function via second-order Taylor expansion:
  $$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
  where $g_i = \partial_{\hat{y}^{(t-1)}} \mathcal{L}(y_i, \hat{y}^{(t-1)})$ represents the first-order gradient, $h_i = \partial_{\hat{y}^{(t-1)}}^2 \mathcal{L}(y_i, \hat{y}^{(t-1)})$ denotes the second-order Hessian, $T$ is the leaf count, and $\gamma, \lambda$ represent tree complexity regularization terms.
- **Suitability for Regression:** The industry standard for tabular data. Sequentially corrects residual errors of prior iterations while explicitly penalizing model complexity.
- **Advantages:** Second-order gradient optimization ensures fast convergence; shrinkage learning rate prevents overfitting; native column/row subsampling decorrelates splits; hardware-optimized cache-aware split finding.
- **Limitations:** Substantial hyperparameter search space (learning rate, max depth, subsample ratios, regularization lambdas); vulnerable to noisy label outliers if learning rate is set too high.

### 9.6 K-Nearest Neighbors Regression (KNN)
- **Mechanism:** KNN is a non-parametric, lazy-learning, instance-based regression technique. Given a test query $\mathbf{x}_q$, it identifies the $K$ closest training instances $\mathcal{N}_K(\mathbf{x}_q)$ according to Minkowski/Euclidean metric distance:
  $$d(\mathbf{x}_q, \mathbf{x}_i) = \left( \sum_{j=1}^d |x_{q,j} - x_{i,j}|^p \right)^{\frac{1}{p}}$$
  In distance-weighted KNN, the prediction is the inverse-distance weighted average of the neighbor target values:
  $$\hat{y}(\mathbf{x}_q) = \frac{\sum_{i \in \mathcal{N}_K(\mathbf{x}_q)} w_i y_i}{\sum_{i \in \mathcal{N}_K(\mathbf{x}_q)} w_i}, \quad w_i = \frac{1}{d(\mathbf{x}_q, \mathbf{x}_i)}$$
- **Suitability for Regression:** Highly effective when the underlying data manifold exhibits strong local spatial autocorrelation (e.g., adjacent houses in identical neighborhoods share similar values).
- **Advantages:** Zero training phase ($\mathcal{O}(1)$ training complexity); no structural distribution assumptions; completely intuitive local geometric interpretation.
- **Limitations:** High inference time complexity $\mathcal{O}(n_{\text{train}} \cdot d)$; severe degradation in high dimensions (curse of dimensionality); high storage requirements.

---

## 10. Model Hyperparameters & Configuration
To ensure a balanced and fair comparative framework, hyperparameters were chosen based on standard industry defaults, data volume, and practical convergence constraints:

| Algorithm | Scaler | Hyperparameter Key | Value | Technical Justification |
|:---|:---:|:---|:---:|:---|
| **Linear Regression** | `StandardScaler` | `fit_intercept` | `True` | Standard OLS closed-form formulation |
| **Decision Tree** | None | `max_depth`<br>`min_samples_split`<br>`min_samples_leaf`<br>`random_state` | `12`<br>`10`<br>`4`<br>`42` | Pruning depth constraint to mitigate extreme single-tree variance |
| **Random Forest** | None | `n_estimators`<br>`max_depth`<br>`min_samples_split`<br>`min_samples_leaf`<br>`n_jobs` | `100`<br>`18`<br>`4`<br>`2`<br>`-1` | Sufficient ensemble capacity (100 trees) with multi-core parallelism |
| **Neural Network** | `StandardScaler` | `hidden_layer_sizes`<br>`activation`<br>`solver`<br>`max_iter`<br>`early_stopping`<br>`n_iter_no_change` | `(64, 32)`<br>`relu`<br>`adam`<br>`350`<br>`True`<br>`12` | Two-layer feedforward architecture with early stopping regularization |
| **XGBoost** | None | `n_estimators`<br>`learning_rate`<br>`max_depth`<br>`subsample`<br>`colsample_bytree` | `150`<br>`0.08`<br>`6`<br>`0.8`<br>`0.8` | Conservative learning rate with stochastic row and feature subsampling |
| **KNN** | `StandardScaler` | `n_neighbors`<br>`weights`<br>`metric`<br>`p` | `9`<br>`distance`<br>`minkowski`<br>`2` | Inverse-distance weighting over 9 neighbors using Euclidean metric |

---

## 11. Evaluation Metrics with Mathematical Formulations
All models are evaluated on the identical test fold $\mathbf{y} \in \mathbb{R}^n$ ($n = 4,128$) against predictions $\hat{\mathbf{y}} \in \mathbb{R}^n$, with test mean denoted as $\bar{y} = \frac{1}{n} \sum_{i=1}^n y_i$.

### 1. Mean Absolute Error (MAE)
Quantifies the average absolute magnitude of prediction errors, providing a linear penalty that is robust to outliers:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

*Units:* Exact target scale ($\$100,000\text{s}$).

### 2. Mean Squared Error (MSE)
Measures the expected value of the squared errors, heavily penalizing large deviations due to quadratic exponentiation:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

*Units:* Squared target units ($\$100,000\text{s}^2$).

### 3. Root Mean Squared Error (RMSE)
The square root of MSE, restoring error magnitude to the original target dimensionality while maintaining quadratic sensitivity to outliers:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

*Units:* Target scale ($\$100,000\text{s}$).

### 4. Coefficient of Determination ($R^2$ Score)
Measures the proportion of target variance explained by the model relative to a baseline horizontal predictor predicting $\bar{y}$:

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$$

*Scale:* Dimensionless; $R^2 = 1.0$ indicates perfect prediction, $R^2 = 0.0$ matches baseline mean prediction, and $R^2 < 0.0$ indicates performance worse than the sample mean.

---

## 12. Experimental Results & Comparative Performance Table

The experimental pipeline was executed end-to-end. Below is the definitive performance table derived from the actual execution of all six trained models on the California Housing test partition:

| Model Rank | Algorithm | MAE ($\$100k$) | MSE | RMSE ($\$100k$) | $R^2$ Score | Training Time (s) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 🥇 **1** | **XGBoost Regression** | **0.3091** | **0.2163** | **0.4650** | **0.8350** | **0.77s** |
| 🥈 **2** | **Random Forest Regression** | **0.3277** | **0.2558** | **0.5057** | **0.8048** | 7.34s |
| 🥉 **3** | **Neural Network (MLPRegressor)** | **0.3533** | **0.2745** | **0.5239** | **0.7905** | 9.88s |
| 4 | **Decision Tree Regression** | 0.4171 | 0.3953 | 0.6287 | 0.6983 | 0.25s |
| 5 | **K-Nearest Neighbors (KNN)** | 0.4375 | 0.4191 | 0.6474 | 0.6801 | 0.07s |
| 6 | **Linear Regression (OLS)** | 0.5332 | 0.5559 | 0.7456 | 0.5758 | **0.07s** |

*Note: In monetary terms, XGBoost achieves an average error of approximately $\$30,907$ ($\text{MAE} = 0.3091 \times \$100,000$), compared to $\$53,320$ for Linear Regression.*

![Model Comparison Multi-Panel](results/figures/model_comparison.png)

![Grouped Metrics Comparison](results/figures/metrics_comparison.png)

---

## 13. Actual vs. Predicted Visual Analysis
The relationship between ground truth values ($x$-axis) and model predictions ($y$-axis) is illustrated across all six algorithms. A perfect regression model maps all points onto the $45^\circ$ reference line ($y = x$).

````carousel
![Actual vs Predicted XGBoost](results/figures/actual_vs_predicted_xgboost.png)
<!-- slide -->
![Actual vs Predicted Random Forest](results/figures/actual_vs_predicted_random_forest.png)
<!-- slide -->
![Actual vs Predicted Neural Network](results/figures/actual_vs_predicted_neural_network.png)
<!-- slide -->
![Actual vs Predicted Decision Tree](results/figures/actual_vs_predicted_decision_tree.png)
<!-- slide -->
![Actual vs Predicted KNN](results/figures/actual_vs_predicted_knn.png)
<!-- slide -->
![Actual vs Predicted Linear Regression](results/figures/actual_vs_predicted_linear_regression.png)
````

### Detailed Scatter Diagnostic Findings:
1. **XGBoost ($R^2 = 0.8350$):** Points tightly cluster along the diagonal reference line across the entire range ($0.5 \le y \le 4.5$). The dispersion bandwidth is narrower than all competing algorithms. The model shows an expected plateau around $y = 5.0$, corresponding to the dataset's artificial upper censoring cap.
2. **Random Forest ($R^2 = 0.8048$):** Tightly clustered along the diagonal, showing slight variance expansion at higher house valuations. Because Random Forest predictions are bounded by the training range, it consistently predicts below 5.0 for capped properties.
3. **Neural Network ($R^2 = 0.7905$):** Generates a smooth, continuous point cloud centered along the diagonal. A slight upward curvature at the lower end ($y < 1.0$) indicates minor boundary compression.
4. **Decision Tree ($R^2 = 0.6983$):** Displays visible horizontal banding artifacts, a characteristic symptom of piecewise-constant step predictions where multiple test queries fall into identical terminal leaves.
5. **K-Nearest Neighbors ($R^2 = 0.6801$):** Greater dispersion than tree ensembles, showing a regression-to-the-mean tendency on extreme values.
6. **Linear Regression ($R^2 = 0.5758$):** High scatter dispersion. In several instances, the linear hyperplane projects predictions below zero ($\hat{y} < 0$), violating physical domain bounds.

---

## 14. Residual Diagnostic Analysis
Residual plots graph prediction errors $e_i = (y_i - \hat{y}_i)$ against predicted values $\hat{y}_i$. An ideal residual distribution exhibits homoscedasticity: constant variance centered at $\mu = 0$ with no systematic curves or patterns.

![Residual Comparison](results/figures/residual_comparison.png)

### Diagnostic Insights:
- **XGBoost:** Exhibits the tightest residual distribution around zero ($\sigma = 0.465$, $\mu = +0.003$), confirming balanced errors across varying price ranges.
- **Random Forest:** Displays symmetrical, low-variance residuals ($\sigma = 0.506$, $\mu = -0.004$) with minimal heteroscedastic fan-out.
- **Neural Network:** Well-centered residuals ($\mu = -0.010$, $\sigma = 0.524$), with slight residual inflation for predictions above $\$4.0$.
- **Linear Regression:** Exhibits severe heteroscedasticity and non-linear drift. At lower predicted values, residuals show pronounced positive drift, while at high values, strong negative residuals reflect the linear model's failure to capture localized non-linear price appreciation.
- **Upper Boundary Censoring Artifact:** All six models exhibit a linear downward diagonal trace in the upper quadrant, corresponding to the dataset's artificial price cap at $\$5.00001$.

---

## 15. In-Depth Strengths and Limitations of Each Algorithm

### 1. Linear Regression (OLS)
- **Strengths:** 
  - Closed-form solution with low computational cost (trained in $0.07$s).
  - Explicit coefficient interpretability: each weight $\beta_j$ directly represents the marginal rate of substitution.
  - Zero hyperparameter tuning required.
- **Limitations:**
  - Strict linearity assumption ($R^2 = 0.5758$).
  - Vulnerable to multicollinearity between spatial coordinates and room ratios.
  - Generates physically implausible negative house value predictions.

### 2. Decision Tree Regression
- **Strengths:**
  - Naturally captures non-linear step-function relationships without feature transformations ($R^2 = 0.6983$).
  - Invariant to monotonic feature scaling.
  - High interpretability via tree decision rules.
- **Limitations:**
  - High variance: small perturbations in training data alter tree topology.
  - Piecewise-constant output leads to discontinuous, stepped predictions.
  - Lower accuracy than ensemble methods.

### 3. Random Forest Regression
- **Strengths:**
  - Substantial variance reduction through bootstrap aggregation ($R^2 = 0.8048$).
  - Highly robust to outliers and noisy training labels.
  - Low hyperparameter sensitivity compared to gradient boosting.
- **Limitations:**
  - Substantial memory footprint for 100 deep trees.
  - Longer training duration ($7.34$s vs. $0.77$s for XGBoost).
  - Incapable of out-of-distribution extrapolation beyond training targets.

### 4. Neural Network Regression (MLPRegressor)
- **Strengths:**
  - Universal approximation capability: models smooth, high-dimensional manifolds ($R^2 = 0.7905$).
  - Scales efficiently to larger datasets via mini-batch SGD.
  - Generates continuous, smooth prediction surfaces.
- **Limitations:**
  - Longest training duration ($9.88$s) among tested algorithms.
  - Requires strict feature scaling; performance degrades on unnormalized inputs.
  - Opaque internal representations ("black box").

### 5. Extreme Gradient Boosting (XGBoost)
- **Strengths:**
  - **Best overall performance:** Highest $R^2$ ($0.8350$) and lowest error metrics ($\text{RMSE} = 0.4650$, $\text{MAE} = 0.3091$).
  - Second-order Taylor optimization delivers fast convergence ($0.77$s training).
  - Built-in L1/L2 leaf regularization prevents overfitting.
  - Native handling of non-linear spatial interactions.
- **Limitations:**
  - Requires careful hyperparameter tuning (`learning_rate`, `max_depth`, subsampling ratios).
  - Can overfit on noisy outliers if learning rates are set too aggressively.

### 6. K-Nearest Neighbors Regression (KNN)
- **Strengths:**
  - No training phase; directly leverages local neighborhood spatial similarity.
  - Simple, non-parametric geometric formulation ($R^2 = 0.6801$).
  - Distance weighting rewards closer geographic and economic neighbors.
- **Limitations:**
  - Degrades in high-dimensional feature spaces (curse of dimensionality).
  - High inference complexity: computes distances against all 16,512 training instances per test sample.
  - Strictly requires feature scaling.

---

## 16. Best Performing Algorithm: Detailed Architectural Analysis

Based on the empirical experimental results, **XGBoost Regression is definitively identified as the best-performing algorithm**, achieving:
- **$R^2$ Score:** `0.8350` (Explaining $83.50\%$ of California housing price variance)
- **RMSE:** `0.4650` ($\approx \$46,504$ average standard deviation)
- **MAE:** `0.3091` ($\approx \$30,907$ mean absolute error)
- **Training Duration:** `0.77` seconds

### Why XGBoost Outperforms Competing Architectures on Tabular Data:

```mermaid
flowchart LR
    A["Input Tabular Features\n(Spatial, Economic, Demographic)"] --> B["Second-Order Taylor Loss\n(Gradient g_i + Hessian h_i)"]
    B --> C["Exact Greedy Split Finding\nOptimal Gain Criterion"]
    C --> D["Shrinkage & Regularization\n(eta=0.08, gamma, lambda)"]
    D --> E["Sequential Error Correction\nAdditive Ensemble of Trees"]
    E --> F["State-of-the-Art Accuracy\n(R² = 0.8350, RMSE = 0.4650)"]
```

1. **Second-Order Taylor Loss Optimization:** Unlike standard gradient boosting which relies solely on first-order residuals (pseudo-residuals), XGBoost incorporates both the first derivative (gradient $g_i$) and second derivative (Hessian $h_i$) of the objective loss function:
   $$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$
   This quadratic approximation provides adaptive step sizing and substantially faster convergence.

2. **Capturing Non-Linear Spatial Manifolds:** Real estate values are heavily influenced by geographic micro-locations. Coastal latitude/longitude combinations demand localized price premiums that a single linear plane cannot approximate. XGBoost recursively partitions latitude and longitude into regional bounding boxes, modeling non-linear geographic price surfaces.

3. **Built-in Regularization Against Overfitting:** The objective includes an explicit complexity penalty $\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum w_j^2$. This restricts tree depth and shrinks leaf weights, preventing the model from fitting localized outliers that degraded the standalone Decision Tree ($R^2 = 0.6983$).

4. **Shrinkage and Stochastic Feature Subsampling:** By applying a conservative shrinkage factor ($\eta = 0.08$) and evaluating random sub-samples of features (`colsample_bytree = 0.8`) and rows (`subsample = 0.8`), individual trees remain decorrelated, yielding an ensemble with lower overall variance.

---

## 17. Conclusion and Future Directions
This study evaluated six representative machine learning regression algorithms on the California Housing dataset under a rigorous 80:20 train-test partition with leakage-free preprocessing.

### Key Conclusions:
1. **Model Hierarchy:** Predictive accuracy follows the ranking:
   $$\text{XGBoost } (0.8350) > \text{Random Forest } (0.8048) > \text{Neural Network } (0.7905) > \text{Decision Tree } (0.6983) > \text{KNN } (0.6801) > \text{Linear Regression } (0.5758)$$
2. **Tabular Domain Dominance:** Tree ensemble architectures—specifically gradient boosting—demonstrate clear advantages on tabular datasets with mixed spatial and economic features, outperforming neural networks and linear models in both accuracy and training efficiency.
3. **Linear Model Limitations:** The linear baseline ($R^2 = 0.5758$) highlights that housing valuation contains spatial and interaction dependencies that cannot be adequately modeled by hyperplanes without manual feature engineering.
4. **Efficiency vs. Accuracy:** XGBoost trained in $0.77$ seconds while outperforming the two-layer neural network ($9.88$s) and Random Forest ($7.34$s), demonstrating strong computational efficiency.

### Future Work:
- **Spatial Feature Engineering:** Incorporate explicit Haversine distance metrics to major coastal metropolitan centers (San Francisco, Los Angeles, San Diego) to enrich geospatial signals.
- **Hyperparameter Optimization:** Apply Bayesian Optimization (`Optuna`) to fine-tune XGBoost's regularization ($\alpha, \lambda$) and tree architecture parameters.
- **Deep Tabular Architectures:** Benchmark against dedicated tabular deep learning models, such as TabNet or FT-Transformer.

---

## 18. References
1. Pace, R. K., & Barry, R. (1997). *Sparse spatial autoregressions*. Statistics & Probability Letters, 33(3), 291-297.
2. Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794.
3. Breiman, L. (2001). *Random forests*. Machine Learning, 45(1), 5-32.
4. Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
5. Friedman, J. H. (2001). *Greedy function approximation: A gradient boosting machine*. Annals of Statistics, 29(5), 1189-1232.
6. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
7. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. Springer Series in Statistics.

---
**© 2026 Atif Ali Shah (FA23-BAI-004)**. Department of Computer Science & Artificial Intelligence.
