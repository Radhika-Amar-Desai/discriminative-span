# Experiments And Discussion

## Experiment One: Is vector arithmetic really applicable for images?

In NLP we observe phenomena such as

```
king − man + woman ≈ queen
```

This suggests that semantic relationships between words can be represented as **consistent vector directions** in embedding space. A natural question is whether a similar phenomenon exists for visual representations learned by modern deep models.

Specifically, we ask:

> If a consistent visual transformation is applied to many images, do the corresponding embedding differences align to form a consistent direction in representation space?

---

### Experiment Design

To test this hypothesis, we construct a controlled **toy synthetic dataset**.

1. We begin with a set of natural images (source images).
2. For every image, we create a transformed version by **superimposing a red patch in the top-right corner**.
3. This produces paired samples:

```
(source image) → (image + red patch)
```

Importantly, the transformation applied to every image is **identical**.

Next, we extract embeddings for both source and transformed images using three different visual representation models:

- DINOv2 (self-supervised vision foundation model)
- ResNet18 (supervised CNN backbone)
- CLIP (contrastively trained vision-language model)

For each image pair we compute a **difference vector**

$$
\Delta_i = f(x_i^{target}) - f(x_i^{source})
$$

where $f(\cdot)$ is the embedding function.

If vector arithmetic holds for images, then all difference vectors should represent the same transformation and therefore **align strongly in embedding space**.

---

### Metrics Used To Measure Alignment of Difference Vectors

To quantify alignment between difference vectors we compute several statistics:

#### 1. Pairwise Cosine Similarity

We compute the cosine similarity between all pairs of difference vectors:

$$
\cos(\Delta_i,\Delta_j)
$$

High mean cosine similarity indicates that the vectors point in similar directions.

---

#### 2. Alignment With Mean Transformation

We compute the mean difference vector

$$
\mu = \frac{1}{N}\sum_i \Delta_i
$$

and measure how well each vector aligns with this mean direction.

---

#### 3. PCA Spectrum

We perform PCA on the difference vectors to analyze the **intrinsic dimensionality** of the transformation.

If the transformation behaves like a single consistent direction, most variance should lie in the **first principal component**.

---

#### 4. Residual Transformation Analysis

After subtracting the mean transformation

$$
r_i = \Delta_i - \mu
$$

we compute cosine similarity between the residual vectors.
If the transformation is truly consistent, the residuals should resemble **random noise** with near-zero mean cosine similarity.

---

### Results

#### DINOv2

| Metric                            | Value       |
| --------------------------------- | ----------- |
| Mean cosine similarity            | **0.565**   |
| Alignment with mean vector        | **0.753**   |
| Variance explained by PC1         | **10.2%**   |
| Variance explained by first 3 PCs | **25.2%**   |
| Residual cosine similarity        | **−0.0029** |

---

#### ResNet18

| Metric                            | Value       |
| --------------------------------- | ----------- |
| Mean cosine similarity            | **0.678**   |
| Alignment with mean vector        | **0.824**   |
| Variance explained by PC1         | **10.3%**   |
| Variance explained by first 3 PCs | **23.1%**   |
| Residual cosine similarity        | **−0.0037** |

---

#### CLIP

| Metric                            | Value       |
| --------------------------------- | ----------- |
| Mean cosine similarity            | **0.815**   |
| Alignment with mean vector        | **0.903**   |
| Variance explained by PC1         | **13.9%**   |
| Variance explained by first 3 PCs | **29.8%**   |
| Residual cosine similarity        | **−0.0032** |

---

### Interpretation

The results reveal several important observations.

#### 1. Difference vectors show substantial alignment

Across all models, the mean cosine similarity between difference vectors is significantly positive:

- **DINOv2:** 0.565
- **ResNet18:** 0.678
- **CLIP:** 0.815

This indicates that the transformation induced by adding the red patch is not random in embedding space. Instead, it produces a **consistent directional shift**.

The effect is strongest for CLIP, suggesting that CLIP encodes this visual modification in a particularly coherent way.

---

#### 2. Alignment with the mean transformation is very strong

The cosine similarity between each difference vector and the mean transformation direction is extremely high:

- **DINOv2:** 0.75
- **ResNet18:** 0.82
- **CLIP:** 0.90

This indicates that most difference vectors lie close to a **shared global direction** in embedding space.

---

#### 3. The transformation is not strictly one-dimensional

PCA analysis shows that the first principal component explains only around **10–14%** of the variance.

This means that the transformation is **not perfectly linear**. Instead, it spans several directions in representation space.

This is expected because the red patch interacts with:

- background textures
- image brightness
- local feature activations
- receptive field structures

which introduces variation across images.

---

#### 4. Residual analysis confirms the presence of a dominant transformation

After subtracting the mean difference vector, residual vectors exhibit near-zero cosine similarity:

```
Residual cosine similarity ≈ 0
```

This indicates that once the dominant transformation direction is removed, the remaining variation behaves like **uncorrelated noise**.

This provides strong evidence that the transformation can be approximated as:

$$
\Delta_i \approx \mu + \epsilon_i
$$

where $\mu$ represents the global transformation and $\epsilon_i$ is image-dependent noise.

---

### Conclusion

This experiment demonstrates that **vector arithmetic partially extends to visual embeddings**.

A consistent visual modification applied across many images induces a **coherent directional shift in representation space**. While the transformation is not perfectly one-dimensional, the strong alignment observed across models suggests that modern visual representations encode certain visual changes as approximately **linear operators** in embedding space.

Among the models tested, CLIP exhibits the strongest directional consistency, likely due to its contrastive training objective which encourages semantically structured embedding spaces.

These findings suggest that difference-vector analysis can serve as a useful tool for studying how visual concepts are represented in deep neural embeddings but it the very property of vector arithmetic that it relies on is only partially true.

---

## Experiment Two: Feature-Weighted Embedding Space

In Experiment One, we analyzed the consistency of difference vectors directly in the **original embedding space** produced by the models.

However, not all embedding dimensions contribute equally to distinguishing between the **source images** and the **transformed images**. Many dimensions may contain irrelevant information or noise unrelated to the transformation.

This motivates a second experiment:

> If we identify and emphasize the features most responsible for distinguishing source and transformed images, does the transformation become more linear and consistent in representation space?

To investigate this, we introduce a **feature-weighted embedding space** derived from a sparse linear classifier.

---

### Experiment Design

We use the same **toy dataset preparation strategy** as in Experiment One.

1. We begin with a set of natural images (source images).
2. For every image, we create a transformed version by **superimposing a red patch in the top-right corner**.
3. This produces paired samples:

```
(source image) → (image + red patch)
```

Importantly, the transformation applied to every image is **identical**.

Next, we extract embeddings for both source and transformed images using three different visual representation models:

- **DINOv2** (self-supervised vision foundation model)
- **ResNet18** (supervised CNN backbone)
- **CLIP** (contrastively trained vision-language model)

---

### Learning Feature Importance

We train a **logistic regression classifier** to distinguish between source and transformed images.

The classifier has the form

$$
w^T x + b = 0
$$

where

- $x$ is the embedding vector
- $w$ is the learned weight vector
- $b$ is the bias term

We apply **L1 regularization**, which encourages **sparsity in the weight vector**. As a result, only a small subset of embedding dimensions receive non-zero weights.

The weight vector $w$ therefore provides **feature importance scores**, indicating which embedding dimensions are most responsible for detecting the transformation.

---

### Constructing a Feature-Weighted Embedding Space

Using the learned weight vector, we transform the embedding space by scaling each feature dimension according to its importance.

This produces a **scaled embedding space** where irrelevant dimensions are suppressed and important dimensions are emphasized.

In this transformed space, we compute difference vectors for each image pair:

$$
\Delta_i = f(x_i^{target}) - f(x_i^{source})
$$

where $f(\cdot)$ now represents the **feature-weighted embedding function**.

If the transformation is primarily captured by the important features, we expect the difference vectors to become **more aligned** in this weighted representation space.

---

### Results

### DINOv2 (Feature-Weighted Space)

| Metric                            | Value        |
| --------------------------------- | ------------ |
| Active dimensions                 | **39 / 384** |
| Mean cosine similarity            | **0.855**    |
| Alignment with mean vector        | **0.925**    |
| Variance explained by PC1         | **30.3%**    |
| Variance explained by first 3 PCs | **52.9%**    |
| Residual cosine similarity        | **−0.0019**  |

---

### ResNet18 (Feature-Weighted Space)

| Metric                            | Value        |
| --------------------------------- | ------------ |
| Active dimensions                 | **14 / 512** |
| Mean cosine similarity            | **0.986**    |
| Alignment with mean vector        | **0.993**    |
| Variance explained by PC1         | **63.6%**    |
| Variance explained by first 3 PCs | **83.1%**    |
| Residual cosine similarity        | **−0.0033**  |

---

### CLIP (Feature-Weighted Space)

| Metric                            | Value       |
| --------------------------------- | ----------- |
| Active dimensions                 | **8 / 512** |
| Mean cosine similarity            | **0.969**   |
| Alignment with mean vector        | **0.985**   |
| Variance explained by PC1         | **48.1%**   |
| Variance explained by first 3 PCs | **90.4%**   |
| Residual cosine similarity        | **−0.0032** |

---

### Interpretation

The results reveal several striking patterns compared to Experiment One.

---

#### 1. Sparse features capture most of the transformation

L1 regularization produces extremely **sparse representations** of the transformation:

- **DINOv2:** 39 active dimensions out of 384
- **ResNet18:** 14 active dimensions out of 512
- **CLIP:** 8 active dimensions out of 512

This suggests that the visual transformation introduced by the red patch is captured by **only a small subset of embedding dimensions**.

---

#### 2. Difference vectors become highly aligned

After transforming the embedding space using feature importance weights, the alignment between difference vectors increases dramatically.

Mean cosine similarity becomes:

- **DINOv2:** 0.855
- **ResNet18:** 0.986
- **CLIP:** 0.969

This indicates that once irrelevant dimensions are suppressed, the transformation behaves much more like a **consistent directional shift**.

---

#### 3. The transformation becomes more low-dimensional

PCA analysis shows that a much larger fraction of variance is captured by the first principal component.

For example:

- **ResNet18:** PC1 explains **63.6%** of the variance
- **CLIP:** PC1 explains **48.1%** of the variance

This indicates that in the feature-weighted space, the transformation is **closer to a single dominant direction**.

---

#### 4. Residuals behave like random noise

After subtracting the mean transformation vector, residual cosine similarity remains near zero:

```
Residual cosine similarity ≈ 0
```

This suggests that once the dominant transformation is removed, the remaining variation between difference vectors behaves like **uncorrelated noise**.

---

### Conclusion

Experiment Two demonstrates that **feature weighting dramatically increases the linearity of visual transformations in embedding space**.

By identifying the embedding dimensions most responsible for distinguishing transformed images, we can construct a representation space in which the transformation behaves much more like a **single consistent direction**.

Compared to the original embedding space analyzed in Experiment One, the feature-weighted space exhibits:

- **much stronger alignment between difference vectors**
- **significantly lower intrinsic dimensionality**
- **extreme sparsity in the relevant features**

These findings suggest that visual transformations may often be encoded in **small, specialized subspaces of the embedding representation**, and that identifying these subspaces can reveal hidden linear structure in deep visual representations.

---

## Experiment Three: Feature-Weighted Embedding Space (L2 Regularization)

In Experiment One, we analyzed the consistency of difference vectors directly in the **original embedding space** produced by the models.

However, not all embedding dimensions contribute equally to distinguishing between the **source images** and the **transformed images**. Many dimensions may contain irrelevant information or noise unrelated to the transformation.

This motivates a third experiment:

> If we identify and emphasize the features most responsible for distinguishing source and transformed images, does the transformation become more linear and consistent in representation space?

To investigate this, we introduce a **feature-weighted embedding space** derived from a linear classifier trained with **L2 regularization**.

---

### Experiment Design

We use the same **toy dataset preparation strategy** as in Experiment One.

1. We begin with a set of natural images (source images).
2. For every image, we create a transformed version by **superimposing a red patch in the top-right corner**.
3. This produces paired samples:

```
(source image) → (image + red patch)
```

Importantly, the transformation applied to every image is **identical**.

Next, we extract embeddings for both source and transformed images using three different visual representation models:

- **DINOv2** (self-supervised vision foundation model)
- **ResNet18** (supervised CNN backbone)
- **CLIP** (contrastively trained vision-language model)

---

### Learning Feature Importance

We train a **logistic regression classifier** to distinguish between source and transformed images.

The classifier has the form

$$
w^T x + b = 0
$$

where

- $x$ is the embedding vector
- $w$ is the learned weight vector
- $b$ is the bias term

In this experiment we apply **L2 regularization**, which encourages **smooth weight distributions rather than sparsity**.

As a result, most embedding dimensions receive **non-zero weights**, but their magnitudes reflect their relative contribution to detecting the transformation.

The weight vector `w` therefore provides **continuous feature importance scores** across the embedding dimensions.

---

### Constructing a Feature-Weighted Embedding Space

Using the learned weight vector, we transform the embedding space by scaling each feature dimension according to its importance.

This produces a **scaled embedding space** where dimensions contributing more strongly to the classifier receive greater weight.

In this transformed space, we compute difference vectors for each image pair:

$$
Δᵢ = f(x_targetᵢ) − f(x_sourceᵢ)
$$

where $f(.)$ now represents the **feature-weighted embedding function**.

If the transformation is primarily captured by the classifier-relevant features, we expect the difference vectors to become **more aligned** in this weighted representation space.

---

### Results

### DINOv2 (L2 Feature-Weighted Space)

| Metric                            | Value         |
| --------------------------------- | ------------- |
| Active dimensions                 | **384 / 384** |
| Mean cosine similarity            | **0.758**     |
| Alignment with mean vector        | **0.871**     |
| Variance explained by PC1         | **15.1%**     |
| Variance explained by first 3 PCs | **31.0%**     |
| Residual cosine similarity        | **−0.0025**   |

### ResNet18 (L2 Feature-Weighted Space)

| Metric                            | Value         |
| --------------------------------- | ------------- |
| Active dimensions                 | **512 / 512** |
| Mean cosine similarity            | **0.880**     |
| Alignment with mean vector        | **0.938**     |
| Variance explained by PC1         | **10.8%**     |
| Variance explained by first 3 PCs | **26.0%**     |
| Residual cosine similarity        | **−0.0037**   |

### CLIP (L2 Feature-Weighted Space)

| Metric                            | Value         |
| --------------------------------- | ------------- |
| Active dimensions                 | **512 / 512** |
| Mean cosine similarity            | **0.918**     |
| Alignment with mean vector        | **0.959**     |
| Variance explained by PC1         | **24.4%**     |
| Variance explained by first 3 PCs | **44.4%**     |
| Residual cosine similarity        | **−0.0029**   |

---

### Interpretation

The results reveal several important contrasts with **Experiment Two**, where L1 regularization was used.

### 1. L2 regularization produces dense feature usage

Unlike L1 regularization, which produced **highly sparse representations**, L2 regularization distributes weight across **all embedding dimensions**.

In all models:

- **DINOv2:** 384 / 384 dimensions active
- **ResNet18:** 512 / 512 dimensions active
- **CLIP:** 512 / 512 dimensions active

This indicates that the classifier relies on **a distributed set of features** rather than isolating a small subset of dominant dimensions.

---

### 2. Difference vector alignment improves modestly

Compared to the **original embedding space**, alignment between difference vectors improves moderately.

Mean cosine similarity becomes:

- **DINOv2:** 0.758
- **ResNet18:** 0.880
- **CLIP:** 0.918

However, this improvement is **smaller than the alignment observed with L1 weighting** in Experiment Two.

This suggests that while emphasizing classifier-relevant features helps reveal structure in the transformation, **dense weighting does not isolate the transformation as effectively as sparse feature selection**.

---

### 3. The transformation remains moderately high-dimensional

PCA analysis shows that the transformation still spreads across multiple directions.

For example:

- **ResNet18:** PC1 explains only **10.8%** of the variance
- **DINOv2:** PC1 explains **15.1%**
- **CLIP:** PC1 explains **24.4%**

Compared to Experiment Two, the transformation is **less concentrated into a single dominant direction**.

This suggests that L2 weighting retains many secondary directions that contribute to variation in the difference vectors.

---

### 4. Residual variation remains unstructured

After subtracting the mean transformation vector, the residual cosine similarity remains near zero:

```
Residual cosine similarity ≈ 0
```

This indicates that once the dominant transformation component is removed, the remaining variation behaves like **uncorrelated noise**, similar to what was observed in Experiments One and Two.

---

### Conclusion

Experiment Three demonstrates that **L2-based feature weighting provides a smoother but less selective transformation of the embedding space**.

Compared to L1 regularization:

- L2 weighting distributes importance across **all embedding dimensions**
- The transformation becomes **more consistent than in the original space**, but
- It remains **less concentrated than in the sparse L1-weighted space**

These findings highlight an important distinction between the two approaches:

- **L1 regularization identifies a small subspace that captures the transformation**
- **L2 regularization spreads the transformation signal across many dimensions**

This suggests that while visual transformations may be represented across many embedding features, the **most informative structure often lies in a small subset of dimensions**, which sparse methods are better able to reveal.

---

## Experiment-Four: Evaluating Span Analysis Metric Alignment with Downstream Performance

### Experiment Objective

---

In this experiment, we evaluate whether our diagnostic metric—**explained fraction**—can serve as an indicator of how well a model **generalizes to real test data**, particularly when trained using synthetic data.

The goal is to analyze whether the alignment between the learned decision boundary and the transformation structure of the data reflects **real-world performance**.

Specifically, we aim to answer:

1. Does **explained_fraction** correlate with **test performance on real data**?
2. Which **embedding space** provides the most meaningful diagnostic signal?
3. Can this metric help identify when a model trained on synthetic data has learned a decision boundary that transfers effectively to real-world data?

---

### Metric Definition

---

We solve the linear system:

$$
D^T \alpha = w
$$

and compute the projected vector:

$$
w_{projected} = D^T \alpha
$$

We then define:

$$
rel\_error = \frac{||w - w_{projected}||_2}{||w||_2}
$$

$$
explained\_fraction = 1 - rel\_error
$$

**Interpretation:**

- **explained_fraction** measures how much of the classifier decision boundary lies within the span of the difference vectors.
- Higher values indicate stronger alignment between the learned decision boundary and the transformation structure captured by the dataset.
- This alignment reflects how well the model captures **semantically meaningful variations**, which influences its ability to generalize to real data.

---

### Experiment Design

---

We conduct experiments on three datasets:

- Pneumonia Chest X-ray (CXR)
- Skin Lesion
- Toy Watermark Dataset

For each dataset:

1. **Train classifiers on synthetic data**
   - Architectures:
     - ResNet-18
     - EfficientNet-B0
     - MobileNet-V2

2. **Evaluate performance on real test data**

$$
\text{Test Accuracy}_{real}
$$

3. **Extract the classifier weight vector**

- For each trained model, we extract the final linear classifier weight vector $w$.

4. **Construct the difference matrix $D$**

- Each row of $D$ corresponds to a **pairwise difference vector** between samples.
- These difference vectors are designed to capture **transformation directions** induced by the dataset.

- Dataset-specific construction:
  - **Pneumonia CXR / Skin Lesion**:
    - Differences are computed between semantically meaningful pairs derived from synthetic transformations.
    - These aim to approximate variations between healthy ↔ diseased states or lesion characteristics.
  - **Toy Watermark Dataset**:
    - Differences are computed between clean images and their **watermarked counterparts**.
    - This creates a highly structured and consistent transformation direction (watermark signal), serving as a **controlled setting** where the transformation subspace is explicitly known and low-dimensional.

5. **Solve the linear system**

$$
D^T \alpha = w
$$

using:

- Least Squares
- Ridge Regularization

6. **Compute diagnostic metrics**

- Relative Error
- Explained Fraction

7. **Embedding space analysis**

All computations are repeated across different embedding spaces:

- ResNet-18 (supervised)
- CLIP (multimodal)
- DINOv2 (self-supervised)

8. **Controlled vs Real-World Comparison**

- The **Toy Watermark dataset** serves as a **sanity-check benchmark**:
  - Tests the behavior of the metric when transformation structure is simple and fully captured.
- The **real-world datasets (CXR, Skin Lesion)** test the metric under:
  - Complex, high-dimensional, and imperfectly modeled transformations.

This design allows us to evaluate whether **explained_fraction behaves consistently across both idealized and realistic settings**, and whether it remains a reliable indicator of downstream generalization.

### Evaluation Strategy

- The metric is analyzed at the level of individual models.
- We study the relationship between **explained_fraction** and **test accuracy on real data**.
- Comparisons across models and embedding spaces are used to assess how consistently the metric reflects real-world generalization performance.

---

### Results

---

### 1. Pneumonia CXR

#### 1.1 Downstream Performance

| Model          | Train Acc | Train F1 | Val Acc | Val F1 Score | Test Acc | Test F1 |
| -------------- | --------- | -------- | ------- | ------------ | -------- | ------- |
| ResNet18       | 1.0       | 1.0      | 0.75    | 0.75         | 0.6138   | 0.5704  |
| MobileNetV2    | 1.0       | 1.0      | 0.625   | 0.7273       | 0.7676   | 0.7858  |
| EfficientNetB0 | 1.0       | 1.0      | 0.6875  | 0.7059       | 0.649    | 0.6231  |

#### 1.2 Diagnostic Metrics (Embedding Space Analysis)

| Embedding Model | Solver        | Effective Rank | Relative Error | Explained Fraction | Number of Pairs | Dim |
| --------------- | ------------- | -------------- | -------------- | ------------------ | --------------- | --- |
| resnet18        | Least Squares | 182            | 0.754          | 0.246              | 259             | 512 |
| resnet18        | Ridge         | 182            | 0.8389         | 0.1611             | 259             | 512 |
| resnet18        | NNLS          | 182            | 0.9756         | 0.0244             | 259             | 512 |
| resnet18        | L1            | 182            | 0.9089         | 0.0911             | 259             | 512 |
| clip            | Least Squares | 145            | 0.7341         | 0.2659             | 259             | 512 |
| clip            | Ridge         | 145            | 0.8966         | 0.1034             | 259             | 512 |
| clip            | NNLS          | 145            | 0.9903         | 0.0097             | 259             | 512 |
| clip            | L1            | 145            | 0.9622         | 0.0378             | 259             | 512 |
| dinov2          | Least Squares | 151            | 0.5943         | 0.4057             | 259             | 384 |
| dinov2          | Ridge         | 151            | 0.8221         | 0.1779             | 259             | 384 |
| dinov2          | NNLS          | 151            | 0.9879         | 0.0121             | 259             | 384 |
| dinov2          | L1            | 151            | 0.8939         | 0.1061             | 259             | 384 |

### 2. Skin Lesion Dataset

#### 2.1 Downstream Performance

| Model          | Train Acc | Train F1 | Val Acc | Val F1 | Test Acc | Test F1 |
| -------------- | --------- | -------- | ------- | ------ | -------- | ------- |
| ResNet18       | 0.9787    | 0.9787   | 0.95    | 0.9474 | 0.9302   | 0.9302  |
| EfficientNetB0 | 0.946     | 0.9448   | 1.0     | 1.0    | 0.8953   | 0.8889  |
| MobileNetV2    | 0.9702    | 0.9698   | 0.95    | 0.9474 | 0.907    | 0.9024  |

#### 2.2 Diagnostic Metrics (Embedding Space Analysis)

| Embedding Model | Solver        | Effective Rank | Relative Error | Explained Fraction | Number of Pairs | Dim |
| --------------- | ------------- | -------------- | -------------- | ------------------ | --------------- | --- |
| resnet18        | Least Squares | 203            | 0.34136        | 0.6586             | 352             | 512 |
| resnet18        | Ridge         | 203            | 0.463288       | 0.5367             | 352             | 512 |
| resnet18        | NNLS          | 203            | 0.687487       | 0.3125             | 352             | 512 |
| resnet18        | L1            | 203            | 0.783373       | 0.2166             | 352             | 512 |
| clip            | Least Squares | 200            | 0.206622       | 0.7934             | 352             | 512 |
| clip            | Ridge         | 200            | 0.319543       | 0.6805             | 352             | 512 |
| clip            | NNLS          | 200            | 0.557877       | 0.4421             | 352             | 512 |
| clip            | L1            | 200            | 0.684727       | 0.3153             | 352             | 512 |
| dinov2          | Least Squares | 190            | 0.121814       | 0.8782             | 352             | 384 |
| dinov2          | Ridge         | 190            | 0.356713       | 0.6433             | 352             | 384 |
| dinov2          | NNLS          | 190            | 0.64572        | 0.3543             | 352             | 384 |
| dinov2          | L1            | 190            | 0.699646       | 0.3004             | 352             | 384 |

### 3. Toy Watermark Dataset

#### 3.1 Downstream Performance

| Model          | Train Acc | Train F1 | Val Acc | Val F1 | Test Acc | Test F1 |
| -------------- | --------- | -------- | ------- | ------ | -------- | ------- |
| ResNet18       | 0.9969    | 0.9969   | 1.0     | 1.0    | 1.0      | 1.0     |
| MobileNetV2    | 1.0       | 1.0      | 1.0     | 1.0    | 0.9778   | 0.9778  |
| EfficientNetB0 | 0.9979    | 0.9979   | 1.0     | 1.0    | 1.0      | 1.0     |

#### 3.2 Diagnostic Metrics (Embedding Space Analysis)

| Embedding Model | Solver        | Effective Rank | Relative Error | Explained Fraction | Number of Pairs | Dim |
| --------------- | ------------- | -------------- | -------------- | ------------------ | --------------- | --- |
| resnet18        | Least Squares | 288            | 0.102834       | 0.8972             | 485             | 512 |
| resnet18        | Ridge         | 288            | 0.356454       | 0.6435             | 485             | 512 |
| resnet18        | NNLS          | 288            | 0.711918       | 0.2881             | 485             | 512 |
| resnet18        | L1            | 288            | 0.726766       | 0.2732             | 485             | 512 |
| clip            | Least Squares | 263            | 0.0817881      | 0.9182             | 485             | 512 |
| clip            | Ridge         | 263            | 0.262815       | 0.7372             | 485             | 512 |
| clip            | NNLS          | 263            | 0.473162       | 0.5268             | 485             | 512 |
| clip            | L1            | 263            | 0.5661         | 0.4339             | 485             | 512 |
| dinov2          | Least Squares | 250            | 2.23e09        | 1                  | 485             | 384 |
| dinov2          | Ridge         | 250            | 0.31784        | 0.6822             | 485             | 384 |
| dinov2          | NNLS          | 250            | 0.547462       | 0.4525             | 485             | 384 |
| dinov2          | L1            | 250            | 0.640865       | 0.3591             | 485             | 384 |

### 4. Horses and Zebra

#### 4.1 Downstream Performance

| Model          | Train Acc | Train F1 | Test Acc | Test F1 |
| -------------- | --------- | -------- | -------- | ------- |
| ResNet18       | 0.8071    | 0.8278   | 0.8375   | 0.866   |
| MobileNetV2    | 0.9893    | 0.9892   | 0.9042   | 0.9098  |
| EfficientNetB0 | 0.9841    | 0.9843   | 0.925    | 0.9302  |

#### 4.2 Diagnostic Metrics (Embedding Space Analysis)

| Embedding Model | Solver        | Effective Rank | Relative Error | Explained Fraction | Number of Pairs | Dim |
| --------------- | ------------- | -------------- | -------------- | ------------------ | --------------- | --- |
| resnet18        | Least Squares | 343            | 2.06e09        | 1                  | 1117            | 512 |
| resnet18        | Ridge         | 343            | 0.417489       | 0.5825             | 1117            | 512 |
| resnet18        | NNLS          | 343            | 0.983051       | 0.0169             | 1117            | 512 |
| clip            | Least Squares | 307            | 1.68e09        | 1                  | 1117            | 512 |
| clip            | Ridge         | 307            | 0.380944       | 0.6191             | 1117            | 512 |
| clip            | NNLS          | 307            | 0.946506       | 0.0535             | 1117            | 512 |
| dinov2          | Least Squares | 283            | 9.55e10        | 1                  | 1117            | 384 |
| dinov2          | Ridge         | 283            | 0.329001       | 0.671              | 1117            | 384 |
| dinov2          | NNLS          | 283            | 0.940155       | 0.0598             | 1117            | 384 |

### 5. Apples and Oranges

#### 5.1 Downstream Performance

| Model          | Train Acc | Train F1 | Test Acc | Test F1 | Val Acc | Val F1 |
| -------------- | --------- | -------- | -------- | ------- | ------- | ------ |
| ResNet18       | 0.9583    | 0.9596   | 0.9372   | 0.9366  | 1.0     | 1.0    |
| MobileNetV2    | 0.9774    | 0.9778   | 0.917    | 0.9165  | 1.0     | 1.0    |
| EfficientNetB0 | 0.9869    | 0.9871   | 0.9291   | 0.9275  | 1.0     | 1.0    |

#### 5.2 Diagnostic Metrics (Embedding Space Analysis)

| Embedding Model | Solver        | Effective Rank | Relative Error | Explained Fraction | Number of Pairs | Dim |
| --------------- | ------------- | -------------- | -------------- | ------------------ | --------------- | --- |
| resnet18        | Least Squares | 351            | 1.21e09        | 1                  | 995             | 512 |
| resnet18        | Ridge         | 351            | 0.375139       | 0.6249             | 995             | 512 |
| resnet18        | NNLS          | 351            | 0.679303       | 0.3207             | 995             | 512 |
| resnet18        | L1            | 351            | 0.663738       | 0.3363             | 995             | 512 |
| clip            | Least Squares | 329            | 7.98e10        | 1                  | 995             | 512 |
| clip            | Ridge         | 329            | 0.290121       | 0.7099             | 995             | 512 |
| clip            | NNLS          | 329            | 0.686178       | 0.3138             | 995             | 512 |
| clip            | L1            | 329            | 0.695665       | 0.3043             | 995             | 512 |
| dinov2          | Least Squares | 268            | 9.05e10        | 1                  | 995             | 384 |
| dinov2          | Ridge         | 268            | 0.413526       | 0.5865             | 995             | 384 |
| dinov2          | NNLS          | 268            | 0.790725       | 0.2093             | 995             | 384 |
| dinov2          | L1            | 268            | 0.693928       | 0.3061             | 995             | 384 |

---

### Correlation Analysis

---

#### 1. CLIP

| Solver        | Pearson r | p-value | Spearman ρ | Num Points |
| ------------- | --------- | ------- | ---------- | ---------- |
| l1            | 0.998     | 0.0016  | 0.8        | 4          |
| least_squares | 0.89      | 0.0432  | 0.553      | 5          |
| nnls          | 0.768     | 0.1294  | 0.821      | 5          |
| ridge         | 0.959     | 0.0099  | 0.975      | 5          |

![CLIP_correlation_best_F1_score](.\plots\clip_correlation.png)

#### 2. Resnet-18

| Solver        | Pearson r | p-value | Spearman ρ | Num Points |
| ------------- | --------- | ------- | ---------- | ---------- |
| l1            | 0.837     | 0.1634  | 0.8        | 4          |
| least_squares | 0.847     | 0.0698  | 0.553      | 5          |
| nnls          | 0.64      | 0.2443  | 0.462      | 5          |
| ridge         | 0.962     | 0.0087  | 0.975      | 5          |

![Resnet18_correlation](.\plots\resnet18_correlation.png)

### 3. Dino-V2

| Solver        | Pearson r | p-value | Spearman ρ | Num Points |
| ------------- | --------- | ------- | ---------- | ---------- |
| l1            | 0.993     | 0.0066  | 1.0        | 4          |
| least_squares | 0.937     | 0.019   | 0.803      | 5          |
| nnls          | 0.777     | 0.1217  | 0.821      | 5          |
| ridge         | 0.946     | 0.015   | 0.667      | 5          |

![Dinov2_correlation](.\plots\dinov2_correlation.png)

## Interpretation

### 1. Relationship Between Explained Fraction and Generalization

Across all datasets, a consistent relationship emerges between **explained_fraction** and downstream generalization performance. This relationship is further supported by **explicit correlation analysis across embedding spaces and solvers**.

- **Higher explained_fraction is generally associated with better test performance on real data**, and this trend is quantitatively validated through strong positive correlations.

From the correlation analysis:

- For **CLIP embeddings**, Pearson correlations range from **0.768 to 0.998**, with statistically significant p-values for Least Squares (p ≈ 0.043) and Ridge (p ≈ 0.0099).
- For **ResNet-18 embeddings**, correlations remain consistently high (≈ 0.64–0.962), with Ridge again showing strong significance (p ≈ 0.0087).
- For **DINOv2 embeddings**, correlations are similarly strong (≈ 0.777–0.993), with multiple solvers achieving statistically significant relationships (p < 0.05).

These results provide **direct quantitative evidence** that explained_fraction is strongly associated with generalization performance across datasets.

For the **Pneumonia CXR dataset**:

- Explained fractions remain relatively low (≈ 0.16 – 0.41 under Least Squares).
- Test performance is also limited (≈ 0.57 – 0.79 F1).
- This aligns with the global trend captured in the correlation analysis, indicating **weak alignment between the classifier and transformation span**, and consequently poorer generalization.

For the **Skin Lesion dataset**:

- Explained fractions are significantly higher (≈ 0.54 – 0.88).
- Test performance is correspondingly strong (≈ 0.89 – 0.93 F1).
- This dataset contributes strongly to the observed positive correlations, demonstrating that **better span alignment leads to improved generalization**.

For the **Toy Watermark dataset**:

- Explained fractions approach unity (≈ 0.64 – 1.00).
- Test performance is near-perfect.
- This represents the upper bound of the correlation trend, where **complete alignment yields optimal performance**.

For the **Horses ↔ Zebras** and **Apples ↔ Oranges** datasets:

- Strong performance is observed despite variability in diagnostic metrics.
- Least Squares solutions often saturate (explained_fraction ≈ 1) due to numerical instability.
- This introduces **degenerate high values**, which partially weaken correlation strength for certain solvers (e.g., NNLS), highlighting the importance of stable estimation.

Overall, both qualitative trends and quantitative correlations support the hypothesis that:

> **Alignment between the classifier and the span of transformation vectors is a key factor governing generalization.**

---

### 2. Effect of Embedding Space

The correlation analysis provides deeper insight into the role of embedding space:

- **CLIP embeddings** exhibit the most consistent and strongest correlations (up to r ≈ 0.998), indicating that explained_fraction in CLIP space is highly predictive of downstream performance.
- **DINOv2 embeddings** also show strong correlations, but with slightly more variability across solvers.
- **ResNet-18 embeddings** demonstrate weaker and less stable correlations, particularly for NNLS.

This suggests:

- The effectiveness of the metric depends not on absolute explained_fraction values, but on how well it **tracks performance variation across datasets**.
- **CLIP provides a representation space where transformation-induced differences are most aligned with task-relevant semantics**, resulting in stronger predictive power.
- DINOv2, while powerful, may introduce **redundant or over-complete representations**, leading to less consistent behavior.

Thus:

> **Embedding spaces should be evaluated based on correlation consistency, not raw metric magnitude**, with CLIP emerging as the most reliable diagnostic space.

---

### 3. Least Squares vs Ridge

Correlation results also clarify the role of solver choice:

- **Ridge regression consistently achieves the strongest and most stable correlations across all embedding spaces**:
  - CLIP: r ≈ 0.959 (p ≈ 0.0099)
  - ResNet18: r ≈ 0.962 (p ≈ 0.0087)
  - DINOv2: r ≈ 0.946 (p ≈ 0.015)

- **Least Squares**, while often producing higher explained_fraction values, shows:
  - Slightly weaker or less stable correlations
  - Sensitivity to ill-conditioning (e.g., saturation in natural image datasets)

- **NNLS and L1**:
  - Show more variability due to constraints
  - Still maintain positive correlation trends, but with weaker statistical significance

This indicates:

- The key signal lies in **span alignment**, but
- **Ridge provides the most reliable estimator of this alignment in practice**

Thus:

> While Least Squares is theoretically ideal, **Ridge offers a better trade-off between fidelity and stability**, making it more suitable for practical diagnostics.

---

### 4. Dataset-Specific Behavior

The correlation analysis reinforces dataset-specific observations:

- **Pneumonia CXR**:
  - Low explained_fraction and weak performance align with the lower end of correlation trends
  - Suggests that synthetic transformations fail to capture clinically relevant variation

- **Skin Lesion**:
  - Strong alignment and performance contribute significantly to observed correlations
  - Indicates effective coverage of task-relevant variation

- **Toy Watermark**:
  - Acts as a boundary case where correlation saturates due to near-perfect alignment

- **Horses/Zebras and Apples/Oranges**:
  - Highlight limitations of the metric under numerical instability
  - Emphasize the need for careful solver choice

This demonstrates that:

> **The metric captures meaningful structure when transformations align with task-relevant variation, but can be distorted under degenerate conditions.**

---

### 5. Key Takeaway

- **Explained_fraction is strongly correlated with generalization performance**, as confirmed by both qualitative trends and quantitative correlation analysis.
- It captures a **necessary condition**: the classifier must lie in the span of transformation vectors for effective generalization.
- However:
  - Numerical instability (e.g., Least Squares saturation)
  - Task simplicity (e.g., Toy dataset)
  - Solver constraints

  can affect interpretation.

Thus:

> **Explained_fraction serves as a principled and empirically validated diagnostic metric, but must be interpreted alongside solver behavior and dataset characteristics.**

---

## Computational Complexity

We analyze the time and space complexity of the span-based diagnostic pipeline.

Let:

- $ n $: number of samples
- $ d $: embedding dimension

---

#### 1. Embedding Extraction

- Time: $ O(n \cdot d) $
- Space: $ O(n \cdot d) $

---

#### 2. Difference Matrix Construction

Each sample has a corresponding transformation (e.g., real ↔ synthetic or clean ↔ watermarked), producing one difference vector per sample:

$
D = x_i - x_j
$

Thus, the number of difference vectors is:

$
m = n
$

- Time: $ O(n \cdot d) $
- Space: $ O(n \cdot d) $

---

#### 3. Solving Linear System

We solve:

$
D^T \alpha = w
$

where $ D \in \mathbb{R}^{n \times d} $

Using least squares / ridge:

- Time: $ O(n \cdot d^2) $
- Space: $ O(n \cdot d) $

---

#### 4. Projection Computation

$
w_{projected} = D^T \alpha
$

- Time: $ O(n \cdot d) $
- Space: $ O(d) $

---

#### 5. Metric Computation

- Relative error and explained fraction: $ O(d) $

(negligible)

---

#### 6. Overall Complexity

The dominant step is solving the linear system.

- **Time Complexity:**  
  $
  O(n \cdot d^2)
  $

- **Space Complexity:**  
  $
  O(n \cdot d)
  $

---

### Summary

Due to the one-to-one construction of difference vectors, the method scales **linearly with the number of samples**, making it efficient and suitable for practical evaluation settings while still capturing meaningful transformation structure.

## Final Conclusion

This experiment demonstrates that $explained\_fraction$ is a meaningful diagnostic metric for assessing generalization to real data.

Key takeaways:

1. **Alignment Drives Generalization**  
   Models whose decision boundaries lie within the span of transformation-induced differences achieve better performance on real data.

2. **Embedding Space Matters**  
   The effectiveness of the metric depends heavily on the representation space, with self-supervised embeddings (DINOv2) providing the most reliable signal.

3. **Dataset Structure is Critical**  
   The metric captures whether the synthetic data encodes meaningful variations of the task:
   - High explained_fraction → synthetic data captures true semantic structure
   - Low explained_fraction → synthetic data fails to represent real-world variability

4. **Practical Utility**  
   The metric can be used as a **diagnostic tool** to:
   - Evaluate whether a trained model is likely to generalize well to real data
   - Identify when synthetic data is sufficiently expressive for the task
   - Compare embedding spaces in terms of their ability to capture meaningful transformations

> Overall, **explained_fraction provides a principled way to connect geometric properties of the learned classifier with real-world performance**, enabling deeper insight into when and why models trained on synthetic data succeed or fail.

---

## Experiment-Five: Evaluating Matrix Rank Metric Alignment with Downstream Performance

In this experiment, we evaluate whether our diagnostic metric—**matrix ranks**—is predictive of **downstream performance degradation** when training on synthetic data instead of real data.

We construct a **difference matrix** $D$, capturing variation in the embedding space, and compute its **effective rank** and **stable rank**. These metrics act as proxies for the **diversity and richness** of the synthetic data distribution.

To further assess **task-specific diversity**, we introduce a linear reweighting of the embedding space using a trained classifier of the form $w^T x + b$. We project embeddings using this learned direction and recompute the rank metrics on the transformed representations. This allows us to evaluate whether diversity aligned with the task is better captured after feature reweighting.

Specifically we aim to answer:

1. Does **effective rank** correlate with the real vs synthetic performance gap?
2. Does **stable rank** correlate with the real vs synthetic performance gap?
3. Which **foundational embedding model** yields the most reliable diagnostic signal?
4. What type of **linear scaling** or **feature weighting** yields the most reliable diagnostic signal?

---

### Metric Definition

1. **Effective rank for raw foundational model embeddings**  
   We compute the effective rank of the difference matrix $ D $, which captures how uniformly information is distributed across dimensions.

2. **Stable rank for raw foundational model embeddings**  
   We compute the stable rank of the difference matrix $ D $, which reflects the concentration of energy in the representation.

3. **Logistic regression-based feature reweighting**  
   We train a logistic regression model $ w^T x + b = y $ and obtain feature-weighted embeddings using:
   $
   x' = w^T x
   $
   We then compute effective and stable rank on the transformed representation.

4. **L1-regularized logistic regression (sparse weighting)**  
   We train a logistic regression model with L1 regularization and compute:
   $
   x' = w^T x
   $
   This emphasizes sparse, task-relevant features. Effective and stable rank are computed on the resulting representation.

5. **L2-regularized logistic regression (smooth weighting)**  
   We train a logistic regression model with L2 regularization and compute:
   $
   x' = w^T x
   $
   This distributes importance more smoothly across features. Effective and stable rank are computed on the resulting representation.

---

### Key Intuition

- **Effective rank** measures how evenly variance is distributed across dimensions (i.e., diversity of representation).
- **Stable rank** measures how concentrated the representation is along dominant directions.
- **Linear reweighting** isolates **task-relevant directions**, allowing us to test whether synthetic data preserves diversity specifically along discriminative axes.

---

### Experiment Design

---

We conduct experiments on three datasets:

- Pneumonia Chest X-ray (CXR)
- Skin Lesion
- Toy Watermark Dataset

For each dataset:

1. **Train classifiers** (ResNet-18, EfficientNet-B0, MobileNet-V2) on:
   - Real data
   - Synthetic data

2. **Measure downstream performance**: Test Accuracy and Test F1 Score

3. **Extract classifier parameters**
   - Obtain the learned weight vector $ w $ from the trained classifier.

4. **Extract embeddings**
   - Use a fixed embedding model (e.g., foundational model or CNN backbone) to extract feature representations $ x \in \mathbb{R}^d $ for all samples.

5. **Construct the difference matrix $ D $**
   - For each class, compute pairwise difference vectors between samples: $ D = x_i - x_j$

6. **Compute rank metrics on raw embeddings**
   - Compute:
     - Effective rank (from normalized singular values)
     - Stable rank (from Frobenius and spectral norms)

7. **Task-aware feature reweighting**
   - Reweight embeddings using classifier weights:
     $
x' = \text{diag}(w)\, x
$

8. **Compute rank metrics on reweighted embeddings**
   - Compute effective rank and stable rank for task-aligned representations

9. **Regularized variants**

- Repeat Steps 3–9 using:
  - Logistic regression with L1 regularization
  - Logistic regression with L2 regularization

10. **Model-wise and dataset-wise comparison**

- Compare which:
  - Embedding model
  - Rank metric
  - Reweighting strategy provides the strongest and most consistent correlation with performance degradation.

---

### Results

### 1. Pneumonia CXR

#### 1.1 Downstream Performance

| Model          | Train Acc | Train F1 | Val Acc | Val F1 Score | Test Acc | Test F1 |
| -------------- | --------- | -------- | ------- | ------------ | -------- | ------- |
| ResNet18       | 0.9846    | 0.9848   | 0.75    | 0.75         | 0.6138   | 0.5704  |
| MobileNetV2    | 1.0       | 1.0      | 0.625   | 0.7273       | 0.7676   | 0.7858  |
| EfficientNetB0 | 0.6875    | 0.7059   | 1.0     | 1.0          | 0.649    | 0.6231  |

#### 1.2 Diagnostic Metrics (Embedding Space Analysis)

| embedding_type | model_type | effective_rank | stable_rank        | span_rank | condition_number | min_singular_value | top5_spectrum_ratio |
| -------------- | ---------- | -------------- | ------------------ | --------- | ---------------- | ------------------ | ------------------- |
| raw            | clip       | 139.37346      | 2.396245497276598  | 259       | 180.12538        | 0.13907792         | 0.19078986          |
| raw            | dinov2     | 148.29431      | 2.4859445709728996 | 259       | 217.0961         | 0.86802006         | 0.17124705          |
| raw            | resnet18   | 162.4744       | 3.7707720593912764 | 259       | 98.910164        | 0.942401           | 0.13836034          |

### 2. Skin Lesion

#### 2.1 Downstream Performance

| Model          | Train Acc | Train F1 | Val Acc | Val F1 | Test Acc | Test F1 |
| -------------- | --------- | -------- | ------- | ------ | -------- | ------- |
| ResNet18       | 0.9787    | 0.9787   | 0.95    | 0.9474 | 0.9302   | 0.9302  |
| EfficientNetB0 | 0.946     | 0.9448   | 1.0     | 1.0    | 0.8953   | 0.8889  |
| MobileNetV2    | 0.9702    | 0.9698   | 0.95    | 0.9474 | 0.907    | 0.9024  |

#### 2.2 Diagnostic Metrics (Embedding Space Analysis)

| embedding_type | model_type | effective_rank | stable_rank        | span_rank | condition_number | min_singular_value | top5_spectrum_ratio |
| -------------- | ---------- | -------------- | ------------------ | --------- | ---------------- | ------------------ | ------------------- |
| raw            | clip       | 191.62308      | 2.7906706779925705 | 352       | 309.50665        | 0.18911652         | 0.14693187          |
| raw            | dinov2     | 187.23073      | 2.644770779198627  | 352       | 891.9587         | 0.43162945         | 0.14300323          |
| raw            | resnet18   | 176.91286      | 2.5911922337165154 | 352       | 616.5882         | 0.17375852         | 0.16643688          |

### 3. Toy Watermark Dataset

#### 3.1 Downstream Performance

| Model          | Train Acc | Train F1 | Val Acc | Val F1 | Test Acc | Test F1 |
| -------------- | --------- | -------- | ------- | ------ | -------- | ------- |
| ResNet18       | 0.9969    | 0.9969   | 1.0     | 1.0    | 1.0      | 1.0     |
| MobileNetV2    | 1.0       | 1.0      | 1.0     | 1.0    | 0.9778   | 0.9778  |
| EfficientNetB0 | 0.9979    | 0.9979   | 1.0     | 1.0    | 1.0      | 1.0     |

#### 3.2 Diagnostic Metrics (Embedding Space Analysis)

| embedding_type | model_type | effective_rank | stable_rank       | span_rank | condition_number | min_singular_value | top5_spectrum_ratio |
| -------------- | ---------- | -------------- | ----------------- | --------- | ---------------- | ------------------ | ------------------- |
| raw            | clip       | 239.88252      | 1.621509155073667 | 485       | 3199.1558        | 0.045821033        | 0.1481445           |
| raw            | dinov2     | 249.88858      | 4.274357498251865 | 383       | 50905124.0       | 6.820158e-06       | 0.0951604           |
| raw            | resnet18   | 265.30124      | 2.328042977518009 | 485       | 1494.5557        | 0.18673043         | 0.117795065         |

### 4. Horses And Zebra

#### 4.1 Downstream Performance

| Model          | Train Acc | Train F1 | Test Acc | Test F1 |
| -------------- | --------- | -------- | -------- | ------- |
| ResNet18       | 0.8071    | 0.8278   | 0.8375   | 0.866   |
| MobileNetV2    | 0.9893    | 0.9892   | 0.9042   | 0.9098  |
| EfficientNetB0 | 0.9841    | 0.9843   | 0.925    | 0.9302  |

#### 4.2 Diagnostic Metrics (Embedding Space Analysis)

| embedding_type | model_type | effective_rank | stable_rank       | span_rank | condition_number | min_singular_value | top5_spectrum_ratio |
| -------------- | ---------- | -------------- | ----------------- | --------- | ---------------- | ------------------ | ------------------- |
| raw            | clip       | 307.82202      | 2.419958479637778 | 511       | 75224.125        | 0.0019066995       | 0.111819945         |
| raw            | dinov2     | 263.4468       | 2.003489822839641 | 383       | 57827350.0       | 1.8490324e-05      | 0.1110306           |
| raw            | resnet18   | 320.51462      | 2.384212825656372 | 512       | 166.73672        | 2.1861253          | 0.10322028          |

### 5. Apples And Oranges

#### 5.1 Downstream Performance

| Model          | Train Acc | Train F1 | Test Acc | Test F1 | Val Acc | Val F1 |
| -------------- | --------- | -------- | -------- | ------- | ------- | ------ |
| ResNet18       | 0.9583    | 0.9596   | 0.9372   | 0.9366  | 1.0     | 1.0    |
| MobileNetV2    | 0.9774    | 0.9778   | 0.917    | 0.9165  | 1.0     | 1.0    |
| EfficientNetB0 | 0.9869    | 0.9871   | 0.9291   | 0.9275  | 1.0     | 1.0    |

#### 5.2 Diagnostic Metrics (Embedding Space Analysis)

| embedding_type | model_type | effective_rank | stable_rank        | span_rank | condition_number | min_singular_value | top5_spectrum_ratio |
| -------------- | ---------- | -------------- | ------------------ | --------- | ---------------- | ------------------ | ------------------- |
| raw            | clip       | 318.7456       | 3.2742655747113214 | 511       | 65165.168        | 0.0020456368       | 0.09653478          |
| raw            | dinov2     | 267.67743      | 2.9477293849769057 | 383       | 61640388.0       | 1.643132e-05       | 0.10264921          |
| raw            | resnet18   | 327.89114      | 6.480032488107657  | 512       | 97.95636         | 2.3061233          | 0.08487251          |

---

### Interpretation

This experiment evaluates whether matrix rank–based diagnostics (effective rank and stable rank) can capture loss of diversity in synthetic data representations and whether this correlates with downstream generalization. We extend this analysis by examining the conditioning of the difference matrix \( D \) through its singular value spectrum.

---

#### 1. Conditioning of the Difference Matrix

Across datasets, we observe that the difference matrix \( D \) is often **ill-conditioned**, as indicated by large condition numbers and very small minimum singular values.

- The Toy Watermark dataset and natural image datasets (Horses ↔ Zebras, Apples ↔ Oranges) exhibit **extreme condition numbers** (up to \(10^7\)), with near-zero singular values.
- This indicates that the difference vectors are **highly dependent** and lie in a **low-dimensional subspace**, even though the ambient dimension is large.

In contrast:

- The Pneumonia CXR dataset has **much lower condition numbers (~10^2)** and larger minimum singular values.
- This suggests that variations are more distributed and less structured.

**Key takeaway:**

> High-performing datasets tend to exhibit _low-dimensional but structured transformations_, leading to ill-conditioned \( D \).

---

#### 2. Rank Alone is Not Sufficient

Effective rank varies across datasets, but does not fully explain performance:

- Toy Watermark and natural image datasets have **high effective rank** _and_ strong performance.
- Pneumonia CXR has **moderate rank** but poor performance.

This reveals an important limitation:

> Effective rank measures _how much variation exists_, but not _how that variation is organized_.

Stable rank is even less informative, remaining in a narrow range across all datasets.

---

#### 3. Interplay Between Rank and Conditioning

A key insight is that:

- High-performing datasets often have:
  - **High effective rank**
  - **Severely ill-conditioned \( D \)**

This means:

- Most meaningful variation is captured in a **few dominant directions**
- Remaining dimensions contribute negligible or noisy variation

In other words:

> The data is high-dimensional in representation, but _effectively low-dimensional in structure_.

---

#### 4. Relationship to Generalization

This explains the observed performance patterns:

- **Toy Watermark**:
  - Extremely ill-conditioned
  - Signal lies in a single dominant direction
  - → Near-perfect performance

- **Natural image datasets**:
  - Ill-conditioned but structured transformations
  - → Strong generalization

- **Pneumonia CXR**:
  - Better conditioned but lacks dominant aligned directions
  - → Poor generalization

Thus:

> Generalization depends not on total diversity, but on whether variation aligns with task-relevant directions.

---

#### 5. Role of Embedding Space

Conditioning behavior varies across embeddings:

- **DINOv2**:
  - Often extremely ill-conditioned
  - Indicates strong compression into dominant directions

- **CLIP**:
  - Moderate conditioning
  - May underrepresent fine-grained medical features

- **ResNet18**:
  - Better conditioned but less semantically structured

This suggests that embedding geometry directly influences both rank and conditioning.

---

#### 6. Key Insight

Across all datasets:

> Synthetic data fails not due to lack of diversity, but due to lack of _aligned, low-dimensional structure_.

Rank-based metrics partially capture diversity, but fail to capture alignment. Conditioning analysis reveals that the true structure lies in how variation is distributed across directions.

---

### Conclusion

We evaluated matrix rank–based diagnostics as a proxy for synthetic data quality across multiple datasets and embedding spaces. While effective rank provides some signal about representation diversity, it is insufficient to explain downstream generalization.

Our analysis shows that the difference matrix \( D \) is often **ill-conditioned**, indicating that transformations induced by synthetic data lie in low-dimensional subspaces with strong linear dependencies. Importantly, this ill-conditioning is not a flaw—in many cases, it reflects structured and consistent transformations that support strong performance.

This leads to the following conclusions:

1. **Effective rank alone is not predictive of performance**
   - Similar rank values can correspond to very different outcomes

2. **Stable rank is too coarse to be informative**

3. **Ill-conditioning reveals underlying structure**
   - High condition numbers indicate low-dimensional transformation subspaces

4. **Generalization depends on alignment, not diversity**
   - Useful variation must align with task-relevant directions

Overall, rank-based metrics capture global properties of the data but fail to account for directional structure. This motivates the need for metrics that explicitly measure whether synthetic transformations encode the discriminative signal required for the task.

This insight directly motivates the development of the Discriminative Span metric, which evaluates whether the classifier direction can be reconstructed from the span of data-induced variations.
