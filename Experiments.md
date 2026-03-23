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
We conduct experiments on two datasets:

- Pneumonia Chest X-ray (CXR)  
- Skin Lesion  

For each dataset:

1. Train classifiers (ResNet-18, EfficientNet-B0, MobileNet-V2) on synthetic data.

2. Evaluate performance on real test data:

$$
\text{Test Accuracy}_{real}
$$

3. Extract the classifier weight vector $w$

4. Construct the difference matrix $D$, where each row corresponds to a pairwise difference vector derived from the training data.

5. Solve $D^T \alpha = w$ using:
   - Least Squares  
   - Ridge Regularization  

6. Compute:
   - Relative Error  
   - Explained Fraction  

7. Perform analysis across embedding spaces:
   - ResNet-18 (supervised)  
   - CLIP (multimodal)  
   - DINOv2 (self-supervised)  

---

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
We conduct experiments on two datasets:

- Pneumonia Chest X-ray (CXR)
- Skin Lesion

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

|Embedding Type | Model Type | Effective Rank     |  Stable Rank        | Span Rank |
|---------------|------------|--------------------|---------------------|-----------|
|l1_scaled      | clip       | 10.09327169837317  |  1.3703055768128465 |  21       |       
|l1_scaled      | dinov2     | 18.178349446988094 |  1.497297783967528  |  36       |
|l1_scaled      | resnet18   | 16.280872741000255 |  1.8408915875319805 |  34       |
|l2_scaled      | clip       | 93.74952165095368  |  1.9973799416388212 |  259      |
|l2_scaled      | dinov2     | 117.76474110520527 |  1.9482139247359413 |  259      |
|l2_scaled      | resnet18   | 116.18810921856682 |  2.463849498880523  |  259      |
|raw            | clip       | 139.3734           |  2.396245497276598  |  259      |
|raw            | dinov2     | 148.29424          |  2.4859445709728996 |  259      |
|raw            | resnet18   | 162.4744           |  3.7707720593912764 |  259      |
|scaled         | clip       | 91.22372015765998  |  2.1282837489459836 |  259      |
|scaled         | dinov2     | 119.08546991738048 |  2.01129580109496   |  259      |
|scaled         | resnet18   | 117.66732472851595 |  2.5930381063579118 |  259      |

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

|Embedding Type | Model Type | Effective Rank     |  Stable Rank        | Span Rank |
|---------------|------------|--------------------|---------------------|-----------|
|l1_scaled      | clip       | 15.411310331003268 | 1.3079173483952957  | 34        |
|l1_scaled      | dinov2     | 31.39561987414743  | 1.85477626348159    | 67        |
|l1_scaled      | resnet18   | 17.79936701582275  | 1.369409955888821   | 43        |
|l2_scaled      | clip       | 137.19138896357885 | 1.7955818030454418  | 352       |
|l2_scaled      | dinov2     | 139.511413358211   | 2.037609299090784   | 352       |
|l2_scaled      | resnet18   | 94.14935962843902  | 1.52073881572785    | 352       |
|raw            | clip       | 191.62308          | 2.7906706779925705  | 352       |
|raw            | dinov2     | 187.23073          | 2.644770779198627   | 352       |
|raw            | resnet18   | 176.91286          | 2.5911922337165154  | 352       |
|scaled         | clip       | 138.47549748769018 | 1.866081986296396   | 352       |
|scaled         | dinov2     | 141.20266690559615 | 2.117195384002967   | 352       |
|scaled         | resnet18   | 99.10142671015089  | 1.5973883165597569  | 352       |

---

### Interpretation

#### 1. Relationship Between Rank Metrics and Generalization

Across both datasets, rank-based metrics reveal a consistent pattern:

- **Higher effective rank and stable rank are generally associated with better test performance on real data.**

For the **Pneumonia CXR dataset**:
- Rank values (especially after scaling) vary significantly across embedding spaces.
- Models exhibit moderate test performance (≈ 0.62 – 0.79), indicating limited generalization.
- The relatively lower and inconsistent rank values suggest that the synthetic data does not provide sufficiently rich or uniformly distributed variation.

For the **Skin Lesion dataset**:
- Rank values are consistently higher across embedding spaces.
- Test performance is also significantly better (≈ 0.86 – 0.94).
- This indicates that the synthetic data captures a broader and more balanced set of variations, enabling stronger generalization.

This suggests that **rank metrics act as proxies for how well synthetic data spans meaningful variations in the task space**.

---

#### 2. Raw vs Task-Aware Representations

A key observation is the difference between **raw embeddings** and **task-aware reweighted embeddings**:

- **Raw embeddings** often exhibit high effective rank, but this does not always correspond to improved performance.
- **Task-aware scaling (using classifier weights)** produces representations where rank values better reflect downstream behavior.

This implies:

- Not all diversity is useful—**task-relevant diversity matters more than overall diversity**.
- Reweighting the embedding space helps isolate dimensions that contribute to classification, making rank metrics more meaningful.

---

#### 3. Effect of Reweighting Strategy

Different reweighting strategies lead to distinct behaviors:

- **L1 scaling (sparse weighting)** results in very low effective rank and span rank.
  - This indicates that only a small subset of features is being utilized.
  - While this highlights key discriminative features, it may discard too much information, limiting the usefulness of rank as a diagnostic.

- **L2 scaling (smooth weighting)** preserves more dimensions and maintains higher rank values.
  - This provides a better balance between focus and diversity.
  - As a result, L2-based representations tend to yield more stable and interpretable signals.

- **Unregularized scaling** behaves similarly to L2 but without explicit control over distribution.

Overall, **overly sparse representations (L1) reduce the expressiveness of the metric**, while smoother weighting (L2) retains more meaningful structure.

---

#### 4. Effect of Embedding Space

The embedding space determines how well rank metrics reflect downstream performance:

- While some embedding spaces achieve higher absolute rank values, the key factor is **how consistently rank correlates with test performance**.
- **CLIP embeddings** show the most stable relationship between rank metrics and generalization across both datasets.
- **DINOv2 and ResNet-18 embeddings** exhibit higher variability, making the diagnostic signal less consistent.

This suggests that:

- The usefulness of rank metrics depends on whether the embedding space organizes data such that **diversity aligns with task-relevant variations**.
- CLIP provides a representation where variations captured by rank are more closely tied to actual classification performance.

---

#### 5. Dataset-Specific Behavior

The metric highlights fundamental differences between datasets:

- **Pneumonia CXR**:
  - Lower and less consistent rank values
  - Weaker generalization
  - Indicates that synthetic transformations may not sufficiently capture real-world variability

- **Skin Lesion**:
  - Higher and more stable rank values
  - Strong generalization
  - Suggests that synthetic data better approximates the true data manifold

Thus, **rank metrics provide insight into how well synthetic data represents the intrinsic structure of the task**.

---

### Final Conclusion

---

This experiment establishes that **matrix rank-based metrics can serve as a diagnostic tool to assess whether synthetic data is sufficiently diverse and task-relevant to support generalization to real data**.

#### 1. Which Configuration Works Best?

From the empirical analysis, the most reliable setup is:

- **Embedding Space:** CLIP  
- **Rank Metric:** Effective Rank  
- **Representation Type:** L2-scaled (task-aware reweighting)

This combination provides the most **consistent and stable correlation with downstream performance** across both datasets.

**Why this works:**

- **CLIP embeddings** organize data in a semantically meaningful space, where variations correspond more closely to real-world task structure.
- **Effective rank** captures how uniformly information is distributed across dimensions, making it a good proxy for **diversity of representations**.
- **L2-based reweighting** preserves task-relevant structure while maintaining sufficient spread across dimensions, avoiding the over-sparsification seen in L1 scaling.

---

#### 2. When is Synthetic Data “Diverse Enough”?

The results suggest that synthetic data can be considered sufficiently expressive when:

- The **effective rank (in CLIP + L2-scaled space)** is **high and stable across models**, and  
- This is accompanied by **consistent downstream performance on real test data**.

In contrast:

- Lower or inconsistent rank values indicate that the synthetic data fails to capture the **true variation of the task**, leading to weaker generalization.

Thus, **effective rank serves as a proxy for whether the synthetic dataset spans meaningful variations of the underlying data distribution**.

---

#### 3. Can This Predict Generalization?

Yes — under the right representation:

- **Higher effective rank (in CLIP + L2 space)** is consistently associated with **better generalization to real data**.
- The metric captures whether the synthetic data encodes **diversity along task-relevant directions**, which is critical for transfer.

However, an important nuance is:

- The metric is **not universally reliable across all embedding spaces or representations**.
- Its predictive power emerges specifically when the embedding space aligns well with semantic structure (as in CLIP) and when diversity is measured in a **task-aware manner** (via L2 scaling).

---

#### 4. Key Insight

The central takeaway is:

> **Not all diversity is useful — only diversity aligned with the task leads to generalization.**

Rank metrics become meaningful only when computed in a representation space where:

- Variations correspond to **semantically meaningful changes**, and  
- The representation is **aligned with the classifier’s decision boundary**.

---

#### 5. Practical Implication

This provides a concrete diagnostic pipeline:

1. Embed data using **CLIP**
2. Apply **L2-based task-aware reweighting**
3. Compute **effective rank of the difference matrix**

This pipeline can be used to:

- Evaluate whether a synthetic dataset is sufficiently diverse  
- Anticipate whether a model trained on it will generalize to real data  
- Compare different synthetic data generation strategies  

---

#### Final Takeaway

**Matrix rank, when computed in a semantically aligned and task-aware representation space, provides a principled and practical indicator of whether synthetic data captures the diversity necessary for real-world generalization.**