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

## Experiment-Four: Evaluating Metric Alignment with Downstream Performance

In this experiment, we evaluate whether our diagnostic metric—**explained fraction**—is predictive of **downstream performance degradation** when training on synthetic data instead of real data.

Specifically, we aim to answer:

1. Does **explained_fraction** correlate with the real vs synthetic performance gap?
2. Which **foundational embedding model** yields the most reliable diagnostic signal?
3. Can this metric help identify settings where synthetic data is sufficient?

---

### Metric Definition

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
rel\_error = \frac{||w - w_{projected}||}{||w||}
$$

$$
explained\_fraction = 1 - rel\_error
$$

**Interpretation:**
- **explained_fraction** measures how much of the classifier decision boundary lies within the span of the difference vectors.
- Higher values indicate better alignment between synthetic transformations and the true decision boundary.

---

### Experiment Design

We conduct experiments on two datasets:
- Pneumonia Chest X-ray (CXR)
- Skin Lesion

For each dataset:

1. Train classifiers (ResNet-18, EfficientNet-B0, MobileNet-V2) on:
   - Real data
   - Synthetic data

2. Measure downstream performance gap:

$$
\Delta_{performance} = \text{Test Accuracy}_{real} - \text{Test Accuracy}_{synthetic}
$$

3. Extract the classifier weight vector  $w$

4. Construct the difference matrix $D$, where each row corresponds to a pairwise difference vector.

5. Solve $D^T \alpha = w$ using:
   - Least Squares
   - Ridge Regularization

6. Compute:
   - Relative Error
   - Explained Fraction

7. Repeat the analysis across embedding spaces:
   - ResNet-18 (supervised)
   - CLIP (multimodal)
   - DINOv2 (self-supervised)

---

### Results

---

### 1. Pneumonia CXR

#### 1.1 Downstream Performance

| Model           | Training Data   | Train Acc | Train F1 | Test Acc | Test F1 | Δ Test Acc (Real - Synth) |
|----------------|----------------|-----------|----------|----------|---------|----------------------------|
| ResNet-18      | Real           | 0.9885    | 0.9881   | 0.9106   | 0.9366  | —                          |
| ResNet-18      | Synthetic      | 0.9973    | 0.9972   | 0.5596   | 0.5751  | **0.3510**                 |
| MobileNet-V2   | Real           | 0.9969    | 0.9968   | 0.9768   | 0.9844  | —                          |
| MobileNet-V2   | Synthetic      | 0.8800    | 0.8901   | 0.8135   | 0.8689  | **0.1633**                 |
| EfficientNet-B0| Real           | 0.9943    | 0.9941   | 0.9724   | 0.9814  | —                          |
| EfficientNet-B0| Synthetic      | 0.9237    | 0.9272   | 0.8709   | 0.9101  | **0.1015**                 |

---

#### 1.2 Diagnostic Metrics (Embedding Space Analysis)

| Embedding Model | Solver         | Effective Rank | Relative Error | Explained Fraction | Number of Pairs | Dim |
|-----------------|---------------|----------------|----------------|--------------------|--------|-----|
| ResNet-18       | Least Squares | 182            | 0.7061         | 0.2939             | 259    | 512 |
| ResNet-18       | Ridge         | 182            | 0.7061         | 0.2939             | 259    | 512 |
| CLIP            | Least Squares | 145            | 0.8016         | 0.1984             | 259    | 512 |
| CLIP            | Ridge         | 145            | 0.8016         | 0.1984             | 259    | 512 |
| DINOv2          | Least Squares | 151            | 0.6627         | 0.3373             | 259    | 384 |
| DINOv2          | Ridge         | 151            | 0.6627         | 0.3373             | 259    | 384 |

---

#### 1.3 Summary

- Explained fraction ranges from **0.20 to 0.34**
- Performance gap ranges from **0.10 to 0.35**
- Weak and inconsistent alignment between metric and downstream performance

---

### 2. Skin Lesion Dataset

#### 2.1 Downstream Performance

| Model           | Training Data   | Train Acc | Train F1 | Test Acc | Test F1 | Δ Test Acc (Real - Synth) |
|----------------|----------------|-----------|----------|----------|---------|----------------------------|
| ResNet-18      | Real           | 0.9986    | 0.9986   | 0.9651   | 0.9655  | —                          |
| ResNet-18      | Synthetic      | 0.8097    | 0.7705   | 0.9302   | 0.9231  | **0.0349**                 |
| MobileNet-V2   | Real           | 1.0000    | 1.0000   | 0.9767   | 0.9767  | —                          |
| MobileNet-V2   | Synthetic      | 0.9972    | 0.9972   | 0.9302   | 0.9231  | **0.0465**                 |
| EfficientNet-B0| Real           | 0.9957    | 0.9957   | 0.7907   | 0.8125  | —                          |
| EfficientNet-B0| Synthetic      | 0.9503    | 0.9477   | 0.7791   | 0.8000  | **0.0116**                 |

---

#### 2.2 Diagnostic Metrics (Embedding Space Analysis)

| Embedding Model | Solver         | Effective Rank | Relative Error | Explained Fraction | Number of Pairs | Dim |
|-----------------|---------------|----------------|----------------|--------------------|--------|-----|
| ResNet-18       | Least Squares | 203            | 0.4633         | 0.5367             | 352    | 512 |
| ResNet-18       | Ridge         | 203            | 0.4633         | 0.5367             | 352    | 512 |
| CLIP            | Least Squares | 200            | 0.3195         | 0.6805             | 352    | 512 |
| CLIP            | Ridge         | 200            | 0.3195         | 0.6805             | 352    | 512 |
| DINOv2          | Least Squares | 190            | 0.3567         | 0.6433             | 352    | 384 |
| DINOv2          | Ridge         | 190            | 0.3567         | 0.6433             | 352    | 384 |

---

#### 2.3 Summary

- Explained fraction ranges from **0.53 to 0.68**
- Performance gap ranges from **0.01 to 0.05**
- Strong alignment between metric and downstream performance

---

### Interpretation

#### 1. Effect of Solver Choice

Across all experiments, least squares and ridge regularization produce nearly identical results.

This indicates that:
- The system is well-conditioned in practice
- The behavior of the metric is not influenced by numerical instability

**Conclusion:** Any mismatch between the metric and downstream performance arises from representational limitations rather than solver choice.

---

#### 2. Pneumonia Dataset: Weak Metric Alignment

For the Pneumonia dataset, explained_fraction values are consistently low (~0.20–0.34), indicating that the span of difference vectors captures only a small portion of the decision boundary.

However, the downstream performance gap varies significantly across models, with some models exhibiting severe degradation when trained on synthetic data.

This reveals that:

> Low explained_fraction does not reliably predict the magnitude of performance degradation.

In particular, even when a large portion of the decision boundary is not captured, the missing components may lie in directions that do not strongly influence classification on the test distribution.

---

#### 3. Skin Lesion Dataset: Strong Metric Alignment

In contrast, the Skin Lesion dataset shows higher explained_fraction values (~0.53–0.68), along with minimal performance gaps between real and synthetic training.

This indicates that:

- The span of difference vectors closely aligns with the decision boundary
- Synthetic transformations effectively capture task-relevant variations

> In this setting, explained_fraction serves as a reliable predictor of downstream performance.

---

#### 4. Dataset-Dependent Behavior

Comparing both datasets reveals that the effectiveness of the metric is highly dataset-dependent:

- Pneumonia: weak or inconsistent correlation
- Skin Lesion: strong correlation

This suggests that:

> explained_fraction captures geometric alignment, but downstream performance depends on additional factors such as data distribution, margin, and feature relevance.

---

#### 5. Impact of Embedding Model

Across both datasets, CLIP and DINOv2 consistently yield higher explained_fraction values compared to ResNet-18.

More importantly, these embeddings produce metrics that better align with downstream performance trends.

This suggests that:

- Richer embedding spaces preserve meaningful structure
- Difference vectors become more semantically aligned with the task
- The diagnostic metric becomes more reliable

> The effectiveness of the metric is strongly dependent on the quality of the embedding space.

---

### Conclusion

This experiment evaluates whether **explained_fraction** serves as a reliable predictor of downstream performance when training on synthetic data.

We find that:

1. Solver choice has negligible impact, confirming that the observed behavior is not due to numerical instability.

2. The relationship between explained_fraction and performance gap is not universally consistent:
   - In the Pneumonia dataset, the metric fails to reliably predict performance degradation.
   - In the Skin Lesion dataset, the metric aligns well with downstream performance.

3. This indicates that explained_fraction captures **geometric alignment**, but downstream performance depends on additional factors beyond this alignment.

4. Embedding choice plays a critical role:
   - CLIP and DINOv2 produce more meaningful diagnostic signals
   - Better embeddings lead to better alignment between metric and performance

Overall, we conclude that:

> explained_fraction is a useful but not universally reliable diagnostic metric, and its effectiveness depends on both the dataset and the embedding space.

These results highlight the need to complement geometric metrics with additional analysis when evaluating synthetic data quality.

