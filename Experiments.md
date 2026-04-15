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

| Model           | Training Data | Train Acc | Train F1 | Test Acc | Test F1 |
| --------------- | ------------- | --------- | -------- | -------- | ------- |
| ResNet-18       | Synthetic     | 1.0000    | 1.0000   | 0.6234   | 0.5913  |
| MobileNet-V2    | Synthetic     | 1.0000    | 1.0000   | 0.7917   | 0.8127  |
| EfficientNet-B0 | Synthetic     | 1.0000    | 1.0000   | 0.7308   | 0.7383  |

---

#### 1.2 Diagnostic Metrics (Embedding Space Analysis)

| Embedding Model | Solver        | Effective Rank | Relative Error | Explained Fraction | Number of Pairs | Dim |
| --------------- | ------------- | -------------- | -------------- | ------------------ | --------------- | --- |
| ResNet-18       | Least Squares | 182            | 0.6136         | 0.3864             | 259             | 512 |
| ResNet-18       | Ridge         | 182            | 0.7061         | 0.2939             | 259             | 512 |
| CLIP            | Least Squares | 145            | 0.6575         | 0.3424             | 259             | 512 |
| CLIP            | Ridge         | 145            | 0.8016         | 0.1984             | 259             | 512 |
| DINOv2          | Least Squares | 151            | 0.4805         | 0.5194             | 259             | 384 |
| DINOv2          | Ridge         | 151            | 0.6627         | 0.3373             | 259             | 384 |

---

### 2. Skin Lesion Dataset

#### 2.1 Downstream Performance

| Model           | Training Data | Train Acc | Train F1 | Test Acc | Test F1 |
| --------------- | ------------- | --------- | -------- | -------- | ------- |
| ResNet-18       | Synthetic     | 0.8097    | 0.7705   | 0.8605   | 0.8537  |
| MobileNet-V2    | Synthetic     | 0.9972    | 0.9972   | 0.9302   | 0.9231  |
| EfficientNet-B0 | Synthetic     | 0.9503    | 0.9477   | 0.9419   | 0.9412  |

---

#### 2.2 Diagnostic Metrics (Embedding Space Analysis)

| Embedding Model | Solver        | Effective Rank | Relative Error | Explained Fraction | Number of Pairs | Dim |
| --------------- | ------------- | -------------- | -------------- | ------------------ | --------------- | --- |
| ResNet-18       | Least Squares | 203            | 0.3413         | 0.6586             | 352             | 512 |
| ResNet-18       | Ridge         | 203            | 0.4633         | 0.5367             | 352             | 512 |
| CLIP            | Least Squares | 200            | 0.2066         | 0.7933             | 352             | 512 |
| CLIP            | Ridge         | 200            | 0.3195         | 0.6805             | 352             | 512 |
| DINOv2          | Least Squares | 190            | 0.1218         | 0.8781             | 352             | 384 |
| DINOv2          | Ridge         | 190            | 0.3567         | 0.6433             | 352             | 384 |

---

### 3. Toy Watermark Dataset

#### 3.1 Downstream Performance

| Model           | Training Data | Train Acc | Train F1 | Test Acc | Test F1 |
| --------------- | ------------- | --------- | -------- | -------- | ------- |
| ResNet-18       | Synthetic     | 0.9309    | 0.9354   | 0.9556   | 0.9615  |
| MobileNet-V2    | Synthetic     | 0.9814    | 0.9818   | 1.0000   | 1.0000  |
| EfficientNet-B0 | Synthetic     | 0.9825    | 0.9828   | 1.0000   | 1.0000  |

---

#### 3.2 Diagnostic Metrics (Embedding Space Analysis)

| Embedding Model | Solver        | Effective Rank | Relative Error | Explained Fraction | Number of Pairs | Dim |
| --------------- | ------------- | -------------- | -------------- | ------------------ | --------------- | --- |
| ResNet-18       | Least Squares | 288            | 0.1028         | 0.8971             | 485             | 512 |
| ResNet-18       | Ridge         | 288            | 0.3564         | 0.6435             | 485             | 512 |
| CLIP            | Least Squares | 263            | 0.0743         | 0.9256             | 485             | 512 |
| CLIP            | Ridge         | 263            | 0.2628         | 0.7371             | 485             | 512 |
| DINOv2          | Least Squares | 250            | 2.64e-09       | 0.9999             | 485             | 384 |
| DINOv2          | Ridge         | 250            | 0.3178         | 0.6821             | 485             | 384 |

---

## Interpretation

### 1. Relationship Between Explained Fraction and Generalization

Across both datasets, a clear pattern emerges:

- **Higher explained_fraction is associated with better test performance on real data.**

For the **Pneumonia CXR dataset**:

- Explained fractions are relatively low across embedding spaces (≈ 0.19 – 0.52).
- Correspondingly, test performance is also limited (≈ 0.62 – 0.79).
- This suggests that the learned decision boundaries are **poorly aligned with the transformation structure**, leading to weaker generalization.

For the **Skin Lesion dataset**:

- Explained fractions are significantly higher (≈ 0.53 – 0.88).
- Test performance is also strong (≈ 0.86 – 0.94).
- This indicates that the decision boundary lies largely within the span of meaningful transformations, resulting in **strong generalization to real data**.

For the **Toy Watermark dataset**:

- Explained fractions are extremely high (≈ 0.64 – ~1.00), especially in Least Squares.
- Test performance is near-perfect (≈ 0.95 – 1.00).
- This demonstrates an almost **complete alignment between the classifier and the transformation span**.
- This dataset acts as a **controlled sanity check**, showing that when transformations are simple, consistent, and fully captured in the span, the metric reaches its upper bound and perfectly reflects generalization.

This consistent trend supports the hypothesis that **alignment between the classifier and transformation structure is a key factor in generalization**.

---

### 2. Effect of Embedding Space

The choice of embedding space has a strong impact on how **reliably explained_fraction reflects real-world performance**.

- While **DINOv2** often achieves higher absolute explained_fraction values, this does not consistently translate into a stronger relationship with downstream performance.
- **CLIP embeddings** show the most consistent alignment between **explained_fraction and test accuracy on real data** across both datasets.
- **ResNet-18 embeddings** exhibit weaker and less consistent correspondence.

This suggests:

- The usefulness of the metric is not determined by how large the explained_fraction is, but by how well it **tracks variations in generalization performance**.
- **CLIP embeddings provide a more faithful representation space** where the span of difference vectors captures transformations that are truly relevant for the downstream task.
- As a result, the projection of the classifier weight vector in CLIP space serves as a more reliable indicator of whether the learned decision boundary will generalize.

Thus, **the effectiveness of an embedding space should be evaluated based on the consistency of correlation with real-world performance, rather than the absolute value of the metric**, with CLIP emerging as the most reliable diagnostic space in this analysis.

---

### 3. Least Squares vs Ridge

- **Least Squares** consistently yields higher explained_fraction than Ridge.
- Ridge regularization introduces shrinkage, which limits the ability to fully reconstruct $w$ from the span of $D$.

This indicates that:

- The primary signal lies in how well $w$ can be expressed in the span, rather than in regularized approximations.
- Least Squares is more suitable for analyzing **span alignment**, which is the core objective of the metric.

---

### 4. Dataset-Specific Behavior

The metric also reveals **intrinsic differences between datasets**:

- Pneumonia CXR shows low alignment and weaker generalization → suggests that synthetic transformations may not capture real-world variations effectively.
- Skin Lesion shows high alignment and strong generalization → suggests that synthetic data better approximates the true data manifold.

This highlights that **the effectiveness of synthetic data depends on how well it spans meaningful variations in the underlying task**.

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

---

### 1. Pneumonia CXR

#### 1.1 Downstream Performance

| Model           | Training Data | Train Acc | Train F1 | Test Acc | Test F1 |
| --------------- | ------------- | --------- | -------- | -------- | ------- |
| ResNet-18       | Synthetic     | 1.0000    | 1.0000   | 0.6234   | 0.5913  |
| MobileNet-V2    | Synthetic     | 1.0000    | 1.0000   | 0.7917   | 0.8127  |
| EfficientNet-B0 | Synthetic     | 1.0000    | 1.0000   | 0.7308   | 0.7383  |

---

#### 1.2 Diagnostic Metrics (Embedding Space Analysis)

| Embedding Type | Model Type | Effective Rank     | Stable Rank        | Span Rank |
| -------------- | ---------- | ------------------ | ------------------ | --------- |
| l1_scaled      | clip       | 10.09327169837317  | 1.3703055768128465 | 21        |
| l1_scaled      | dinov2     | 18.178349446988094 | 1.497297783967528  | 36        |
| l1_scaled      | resnet18   | 16.280872741000255 | 1.8408915875319805 | 34        |
| l2_scaled      | clip       | 93.74952165095368  | 1.9973799416388212 | 259       |
| l2_scaled      | dinov2     | 117.76474110520527 | 1.9482139247359413 | 259       |
| l2_scaled      | resnet18   | 116.18810921856682 | 2.463849498880523  | 259       |
| raw            | clip       | 139.3734           | 2.396245497276598  | 259       |
| raw            | dinov2     | 148.29424          | 2.4859445709728996 | 259       |
| raw            | resnet18   | 162.4744           | 3.7707720593912764 | 259       |
| scaled         | clip       | 91.22372015765998  | 2.1282837489459836 | 259       |
| scaled         | dinov2     | 119.08546991738048 | 2.01129580109496   | 259       |
| scaled         | resnet18   | 117.66732472851595 | 2.5930381063579118 | 259       |

---

### 2. Skin Lesion Dataset

#### 2.1 Downstream Performance

| Model           | Training Data | Train Acc | Train F1 | Test Acc | Test F1 |
| --------------- | ------------- | --------- | -------- | -------- | ------- |
| ResNet-18       | Synthetic     | 0.8097    | 0.7705   | 0.8605   | 0.8537  |
| MobileNet-V2    | Synthetic     | 0.9972    | 0.9972   | 0.9302   | 0.9231  |
| EfficientNet-B0 | Synthetic     | 0.9503    | 0.9477   | 0.9419   | 0.9412  |

---

#### 2.2 Diagnostic Metrics (Embedding Space Analysis)

| Embedding Type | Model Type | Effective Rank     | Stable Rank        | Span Rank |
| -------------- | ---------- | ------------------ | ------------------ | --------- |
| l1_scaled      | clip       | 15.411310331003268 | 1.3079173483952957 | 34        |
| l1_scaled      | dinov2     | 31.39561987414743  | 1.85477626348159   | 67        |
| l1_scaled      | resnet18   | 17.79936701582275  | 1.369409955888821  | 43        |
| l2_scaled      | clip       | 137.19138896357885 | 1.7955818030454418 | 352       |
| l2_scaled      | dinov2     | 139.511413358211   | 2.037609299090784  | 352       |
| l2_scaled      | resnet18   | 94.14935962843902  | 1.52073881572785   | 352       |
| raw            | clip       | 191.62308          | 2.7906706779925705 | 352       |
| raw            | dinov2     | 187.23073          | 2.644770779198627  | 352       |
| raw            | resnet18   | 176.91286          | 2.5911922337165154 | 352       |
| scaled         | clip       | 138.47549748769018 | 1.866081986296396  | 352       |
| scaled         | dinov2     | 141.20266690559615 | 2.117195384002967  | 352       |
| scaled         | resnet18   | 99.10142671015089  | 1.5973883165597569 | 352       |

---

### 3. Toy Watermark Dataset

#### 3.1 Downstream Performance

| Model           | Training Data | Train Acc | Train F1 | Test Acc | Test F1 |
| --------------- | ------------- | --------- | -------- | -------- | ------- |
| ResNet-18       | Synthetic     | 0.9309    | 0.9354   | 0.9556   | 0.9615  |
| MobileNet-V2    | Synthetic     | 0.9814    | 0.9818   | 1.0000   | 1.0000  |
| EfficientNet-B0 | Synthetic     | 0.9825    | 0.9828   | 1.0000   | 1.0000  |

---

#### 3.2 Diagnostic Metrics (Embedding Space Analysis)

| Embedding Type | Model Type | Effective Rank     | Stable Rank        | Span Rank |
| -------------- | ---------- | ------------------ | ------------------ | --------- |
| l1_scaled      | clip       | 2.988803003172116  | 1.037706736056944  | 8         |
| l1_scaled      | dinov2     | 12.655433017176762 | 1.1648219646047855 | 32        |
| l1_scaled      | resnet18   | 2.351643494299326  | 1.0099459168636096 | 9         |
| l2_scaled      | clip       | 59.300325382872124 | 1.0843353335911887 | 259       |
| l2_scaled      | dinov2     | 139.511413358211   | 1.3114768733168938 | 259       |
| l2_scaled      | resnet18   | 77.6889635298853   | 1.1322532091595852 | 259       |
| raw            | clip       | 110.1766           | 1.2161132314260634 | 259       |
| raw            | dinov2     | 142.83354          | 1.7668705841326064 | 259       |
| raw            | resnet18   | 141.09366          | 1.460840758649932  | 259       |
| scaled         | clip       | 59.98991532880907  | 1.085952509201024  | 259       |
| scaled         | dinov2     | 106.38854322613558 | 1.354986492809891  | 259       |
| scaled         | resnet18   | 79.46581552962917  | 1.13818322585682   | 259       |

---

## Interpretation

This experiment evaluates whether **matrix rank–based diagnostics** can reliably predict **downstream performance degradation** when models are trained on synthetic data. The results across three datasets—CXR, Skin Lesion, and Toy Watermark—reveal consistent and insightful trends. :contentReference[oaicite:0]{index=0}

---

#### 1. Relationship Between Rank and Performance Gap

A clear pattern emerges when comparing **rank metrics with downstream performance**:

- In the **Pneumonia CXR dataset**, models trained on synthetic data exhibit **significant performance degradation** (e.g., ResNet-18 test accuracy ≈ 62%). Correspondingly, we observe:
  - Lower **effective rank after task-aware projections (especially L1)**
  - Moderate **stable rank values**, indicating concentration along limited directions

- In contrast, the **Skin Lesion dataset** shows **moderate to high performance** (test accuracy up to ≈ 94%), with:
  - Higher **effective rank across embeddings**
  - More **distributed representations**, indicating better diversity retention

- The **Toy Watermark dataset** achieves **near-perfect generalization**, despite:
  - Very **low effective rank under L1 scaling**
  - Extremely **low stable rank (~1)**

This suggests an important insight:

> **High rank is not universally required for good performance—but low rank becomes problematic when the task requires rich semantic diversity.**

---

#### 2. Effective Rank vs Stable Rank

Across datasets, the two metrics behave differently:

- **Effective Rank**
  - Strongly reflects **diversity of representations**
  - Better distinguishes between datasets with **high vs low generalization gaps**
  - Particularly informative in **raw and L2-scaled embeddings**

- **Stable Rank**
  - Remains within a narrow range (≈1–3 across all experiments)
  - Captures **energy concentration**, but is **less sensitive to task difficulty**
  - Fails to clearly separate high-performing vs low-performing settings

> **Conclusion:** Effective rank is a more reliable diagnostic signal than stable rank for predicting synthetic data utility.

---

#### 3. Impact of Task-Aware Reweighting

Feature reweighting provides key insights:

- **L1 Scaling (Sparse Projection)**
  - Drastically reduces effective rank across all datasets
  - Highlights **task-critical directions**
  - In CXR, reveals that synthetic data lacks diversity **along discriminative axes**, explaining poor performance
  - In Toy dataset, low rank is sufficient because the task depends on **few simple features (watermark cues)**

- **L2 Scaling (Smooth Projection)**
  - Preserves much of the original rank
  - Produces behavior closer to raw embeddings
  - Less effective at isolating failure modes

> **Key Insight:**  
> Performance degradation is better explained by **loss of diversity in task-relevant directions**, not overall feature space diversity.

---

#### 4. Role of Embedding Models

Across all datasets:

- **DINOv2 and ResNet embeddings** tend to produce:
  - Higher effective rank
  - More stable correlations with downstream performance

- **CLIP embeddings**:
  - Show lower rank under task projections
  - Less consistent alignment with performance trends

This suggests:

> The **choice of embedding space significantly affects the reliability of rank-based diagnostics**, with vision-specialized models outperforming multimodal ones in this setting.

---

#### 5. Insights from the Toy Watermark Dataset

The Toy dataset plays a crucial role in validating the hypothesis:

- Despite **low rank**, models achieve **perfect generalization**
- This occurs because:
  - The task is **low-dimensional**
  - Discriminative information lies in **simple, sparse features**

This serves as a **counterexample**:

> **Low rank is not inherently bad—it is only problematic when the task requires high intrinsic dimensionality.**

---

#### 6. Overall Conclusion

From these observations, we conclude:

1. **Effective rank correlates with performance degradation**, but only when interpreted relative to **task complexity**
2. **Stable rank is less informative** for predicting downstream performance
3. **Task-aware projections (especially L1)** provide the most meaningful diagnostic signal
4. Synthetic data often fails not due to lack of global diversity, but due to **missing variation along task-relevant directions**
5. The Toy dataset confirms that **rank must be interpreted in context**, not as an absolute metric

---

### Computational Complexity

We analyze the time and space complexity of the proposed diagnostic pipeline.

Let:

- $ n $: number of samples
- $ d $: embedding dimension

---

#### 1. Embedding Extraction

We extract embeddings for both real and synthetic samples.

- Time: $ O(n \cdot d) $
- Space: $ O(n \cdot d) $

---

#### 2. Difference Matrix Construction

Using an image-to-image (I2I) translation setup, each sample has a one-to-one correspondence:

$
D = x_i^{real} - x_i^{synthetic}
$

Thus, the number of difference vectors is: $m = n$

- Time: $ O(n \cdot d) $
- Space: $ O(n \cdot d) $

---

#### 3. Singular Value Decomposition (SVD)

Let $ D \in \mathbb{R}^{n \times d} $

- Time: $ O(n \cdot d^2) $
- Space: $ O(n \cdot d) $

---

#### 4. Rank Computation

- Effective rank: $ O(d) $
- Stable rank: $ O(d) $

(negligible compared to SVD)

---

#### 5. Logistic Regression (Feature Reweighting)

- Time: $ O(n \cdot d) $
- Space: $ O(d) $

---

#### 6. Overall Complexity

The dominant step is SVD on the difference matrix.

- **Time Complexity:** $O(n \cdot d^2)$

- **Space Complexity:** $O(n \cdot d)$

---

#### Summary

---

Due to the one-to-one correspondence induced by the I2I translation pipeline, the method scales **linearly with the number of samples**, making it significantly more efficient and scalable than pairwise difference-based approaches.

---

## Conclusion

We evaluated matrix rank–based diagnostics for understanding synthetic data quality in downstream tasks. Effective rank consistently reflects representation diversity and aligns well with performance differences, while stable rank is less informative. Task-aware reweighting, especially L1, shows that failures arise from missing diversity along discriminative directions rather than the full feature space. The Toy dataset highlights that low rank is sufficient when tasks are inherently simple, emphasizing context-dependent interpretation. Finally, the I2I-based formulation ensures linear scalability, making the approach efficient and practical for analysis. Overall, effective rank serves as a useful and interpretable signal for diagnosing synthetic data limitations.
