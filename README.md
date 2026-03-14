# Knowing the Difference

A geometric framework for diagnosing the usefulness of synthetic data using **difference vectors in representation space**.

---

## Overview

Synthetic data generation has become an increasingly important tool in machine learning, particularly in domains where collecting labeled data is expensive or difficult (for example, medical imaging). Modern image-to-image translation models can generate synthetic samples that simulate how an image from one class might appear if it belonged to another class.

However, a fundamental question remains:

**How can we determine whether synthetic data actually captures the signal needed to learn a good classifier?**

Training and evaluating multiple models to answer this question can be computationally expensive and time-consuming. This project explores an alternative approach based on the **geometry of representation spaces**.

Inspired by the well-known vector arithmetic observed in word embeddings (Word2Vec), where semantic relationships emerge through vector differences:

```
king − man + woman ≈ queen
```

we investigate whether **difference vectors between real and synthetic images encode meaningful semantic transformations in visual representation spaces**.

If synthetic data correctly captures the transformation between classes, these difference vectors should collectively explain the **decision boundary of a classifier trained on real data**.

---

## Geometric Intuition

The core idea can be understood geometrically.

We embed images into a representation space using a pretrained vision model. For each real image `x`, we generate a synthetic counterpart `x'` representing how that image would appear if it belonged to the opposite class.

The **difference vector**

```
d = x' − x
```

represents the transformation applied to the image.

If the synthetic transformation correctly captures the class difference, these vectors should align with the direction separating the classes — the **decision boundary normal `w`**.

---

**Interpretation**

- Difference vectors represent **semantic transformations**
- These vectors form the **difference matrix `D`**
- We test whether these transformations explain the **decision boundary**

---

## Core Idea

Consider a **binary classification task** with classes **A** and **B**.

For each real image belonging to class **A**, we generate a synthetic image representing how the same image would appear if it belonged to class **B** using an image-to-image translation model.

Example:

```
Healthy chest X-ray → synthetic pneumonia X-ray
```

Each pair forms an image pair, and we define the difference vector

```
d = x' − x
```

This vector represents the transformation applied to move the sample from class **A toward class B** in representation space.

Our hypothesis is that these transformation vectors capture the **semantic direction separating the two classes**.

---

## Representation Space

Rather than working with raw images, we extract embeddings using a pretrained foundation model such as:

- CLIP
- DINO / DINOv2
- other vision encoders

Each image is represented as a vector

```
x, x' ∈ ℝ^d
```

where `d` is the embedding dimension.

For each pair we compute

```
d_i = x'_i − x_i
```

Stacking these vectors produces the **difference matrix**

```
D = [ d1
      d2
      ...
      dn ]
```

where

```
D ∈ ℝ^(n × d)
```

and `n` is the number of real–synthetic pairs.

---

## Decision Boundary

To approximate the true class separation, we train a **linear classifier on real data**.

The classifier defines the decision boundary

```
wᵀx + b = 0
```

where

- `w` is the **normal vector of the decision boundary**
- `b` is the **bias term**

Since the true optimal boundary is unknown, we use the boundary learned from real data as a **proxy**.

---

## Hypothesis

If synthetic transformations correctly capture the semantic difference between classes, the decision boundary normal should lie within the **span of the difference vectors**.

Formally:

```
w ≈ Dᵀ α
```

where

- `D` is the difference vector matrix
- `α` are coefficients representing the contribution of each difference vector

We estimate `α` by solving the **least squares problem**

```
min_α || Dᵀα − w ||²
```

This produces

```
w' = Dᵀ α
```

which represents the component of the decision boundary explained by the synthetic transformations.

---

## Explanation Score

To measure how well the difference vectors explain the decision boundary, we compute the **cosine similarity** between `w` and `w'`:

```
Explanation Score =
(w · w') / (||w|| ||w'||)
```

### Interpretation

| Score | Meaning |
|------|------|
| Close to **1** | Synthetic transformations explain the decision boundary |
| Close to **0** | Synthetic data does not capture the true class signal |

This provides a **quantitative measure of synthetic data quality**.

---

## Interpretation

### High Explanation Score

If the difference vectors align well with the decision boundary:

- Synthetic transformations capture the **true semantic direction between classes**
- Models trained on synthetic data should perform **similarly to models trained on real data**

### Low Explanation Score

If alignment is weak:

- Synthetic data fails to represent the **real class separation**
- Models trained on synthetic data will likely perform **significantly worse**

---

## Why This Matters

Evaluating synthetic datasets typically requires **training multiple models**, which is expensive and time-consuming.

This framework offers a **model-agnostic diagnostic tool** that analyzes the geometry of representation spaces to determine whether synthetic data contains the signal needed to learn an accurate classifier.

This allows practitioners to estimate the usefulness of synthetic data **before investing heavily in model development**.

---

## Potential Applications

### Synthetic Data Quality Diagnosis

The primary goal of this work is diagnosing whether synthetic datasets generated through **image-to-image translation** can support high-quality model training.

### Representation Drift

The analysis can be performed using different embedding spaces:

- embeddings from a **foundation model**
- embeddings from a **model being fine-tuned**

Large differences in explanation score may indicate **representation drift during training**.

### Domain Adaptation

The coefficients obtained from the least squares solution indicate the **relative importance of each transformation vector**.

These coefficients could potentially be used to **weight synthetic training samples**, improving alignment with the true decision boundary.

### Coreset Selection

Difference vectors may also be used for **data subset selection**.

Gradient-based coreset selection algorithms such as **GradMatch** could potentially be adapted by replacing gradients with difference vectors to identify representative samples.

---

## Summary

This project investigates whether semantic transformations generated by **image-to-image translation** correspond to meaningful directions in representation space.

By analyzing **difference vectors between real and synthetic image pairs**, we attempt to determine whether synthetic data captures the signal required to learn accurate classifiers.

If successful, this framework provides a **simple geometric method for diagnosing the usefulness of synthetic datasets before expensive model training**.

---

## Status

This repository contains an **ongoing exploratory study** investigating the geometric structure of synthetic transformations in representation spaces.

Experiments and methodology are **actively evolving**.
