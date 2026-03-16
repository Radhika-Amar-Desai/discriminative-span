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

* DINOv2 (self-supervised vision foundation model)
* ResNet18 (supervised CNN backbone)
* CLIP (contrastively trained vision-language model)

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

* **DINOv2:** 0.565
* **ResNet18:** 0.678
* **CLIP:** 0.815

This indicates that the transformation induced by adding the red patch is not random in embedding space. Instead, it produces a **consistent directional shift**.

The effect is strongest for CLIP, suggesting that CLIP encodes this visual modification in a particularly coherent way.

---

#### 2. Alignment with the mean transformation is very strong

The cosine similarity between each difference vector and the mean transformation direction is extremely high:

* **DINOv2:** 0.75
* **ResNet18:** 0.82
* **CLIP:** 0.90

This indicates that most difference vectors lie close to a **shared global direction** in embedding space.

---

#### 3. The transformation is not strictly one-dimensional

PCA analysis shows that the first principal component explains only around **10–14%** of the variance.

This means that the transformation is **not perfectly linear**. Instead, it spans several directions in representation space.

This is expected because the red patch interacts with:

* background textures
* image brightness
* local feature activations
* receptive field structures

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

| Metric | Value |
|------|------|
| Active dimensions | **39 / 384** |
| Mean cosine similarity | **0.855** |
| Alignment with mean vector | **0.925** |
| Variance explained by PC1 | **30.3%** |
| Variance explained by first 3 PCs | **52.9%** |
| Residual cosine similarity | **−0.0019** |

---

### ResNet18 (Feature-Weighted Space)

| Metric | Value |
|------|------|
| Active dimensions | **14 / 512** |
| Mean cosine similarity | **0.986** |
| Alignment with mean vector | **0.993** |
| Variance explained by PC1 | **63.6%** |
| Variance explained by first 3 PCs | **83.1%** |
| Residual cosine similarity | **−0.0033** |

---

### CLIP (Feature-Weighted Space)

| Metric | Value |
|------|------|
| Active dimensions | **8 / 512** |
| Mean cosine similarity | **0.969** |
| Alignment with mean vector | **0.985** |
| Variance explained by PC1 | **48.1%** |
| Variance explained by first 3 PCs | **90.4%** |
| Residual cosine similarity | **−0.0032** |

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

| Metric | Value |
|------|------|
| Active dimensions | **384 / 384** |
| Mean cosine similarity | **0.758** |
| Alignment with mean vector | **0.871** |
| Variance explained by PC1 | **15.1%** |
| Variance explained by first 3 PCs | **31.0%** |
| Residual cosine similarity | **−0.0025** |

### ResNet18 (L2 Feature-Weighted Space)

| Metric | Value |
|------|------|
| Active dimensions | **512 / 512** |
| Mean cosine similarity | **0.880** |
| Alignment with mean vector | **0.938** |
| Variance explained by PC1 | **10.8%** |
| Variance explained by first 3 PCs | **26.0%** |
| Residual cosine similarity | **−0.0037** |

### CLIP (L2 Feature-Weighted Space)

| Metric | Value |
|------|------|
| Active dimensions | **512 / 512** |
| Mean cosine similarity | **0.918** |
| Alignment with mean vector | **0.959** |
| Variance explained by PC1 | **24.4%** |
| Variance explained by first 3 PCs | **44.4%** |
| Residual cosine similarity | **−0.0029** |

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
