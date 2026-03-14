# Knowing_the_difference

This project is an exploratory study analyzing how difference vectors can be exploited particularly when training classifiers with synthetic data generated using image-to-image translation technique on real data.
The project is inspired by the famous vector arthmetic operations in the field of NLP and word2vec where property:
```
king - man + queen = woman
```
stands true.

Here difference vector refers to the difference of vectors of image pairs. An image pair constitutes of a real image, and a synthetic image obtained via image-to-image translation.
Our hypothesis is that these difference vectors influence the decision boundary for the given binary classification task.

For a binary classification task with classes A and B, we take the real data points from class A and apply image to image translation process to obtain corresponding images in class B. Basically if this image
belonged to class B what would it look like. For example, if the given binary classification task is to differentiate between a chest X-ray of a healthy patient vs chest X-ray of a pneumonia patient. We obtain
synthetic images for pneumonia by incorporating consolidation patterns in each image of chest X-rays belonging to the healthy class.

We define ```x' = x + c * w``` where x belongs to class A and x' is synthetic data obtained via image-to-image translation technique. w stands for normal of the ideal decision boundary.

Now a matrix of difference vectors is obtained as D with ```x' - x``` as rows of D. We then solve least square solution Ax = B but with A = D, B = normal of the ideal decision boundary.
Since it is impossible to determine the perfect and ideal decision boundary, we take the decision boundary of model trained on real data as the proxy for the ideal decision boundary.

To simplify things further, we take embeddings from pre-trained model or foundational model to be able to treat x and x' as vectors instead of raw images. In order to determine the ideal decision boundary,
the model used to train the real data points is a linear classifier with decision boundary ```bash wT * x + b = 0```

Finally the equation we are trying to solve becomes:
```D^T * alpha = w ``` where D = difference vector matrix, w = normal to decision boundary.

This is a test to answer if the differnece vectors can form a linear combination that contains the ideal decision boundary. Since foundational model embeddings are being used, we believe that difference vector
symbolize semantic meaning or concepts. So are the concepts represented in my synthetic data capable of representing the actual signal which differentiates the two classes ?

If this is so, there exists a model that when trained on synthetic data performs nearly as well as model trained on real data. If not, then it is almost impossible to find such a model and model trained on
real data will significantly perform better than the model trained on synthetic data.

This allows us to exploit difference vectors to be used as a model agnoistic tool for diagnosis of data quality of synthetic data and get an intution on whether or not it is worth investing in model development
using our synthetic data. In order to assign a quantative score, we calculate the projection error and obtain an explanation fraction.

We try solving: ```D^T * alpha = w``` and obtain a value for ```alpha```. Now we have ```w' = D^T * alpha```.
We calculate ```projection error = w.w' / ||w|| * ||w'||``` and ```explanation fraction = 1 - projection error```

## Playing with the tool

The primary motivation of this mathematical tool remains solving the problem of diagnosing whether or not my synthetic data can help me build a model as good as my real data. However, we also believe that
there are some other potential applications that data scientists may use this tool for.

### Commenting on representation drift

Instead of foundational model embeddings for this analysis, data scientists can perform analysis on both foundational model embeddings and embeddings of model they intend to train or fine tune. For example,
if we plan to finetune pretrained resnet-18. We test on pretrained resnet-18 model embeddings. If it happens, that we have a high score for analysis done using foundational model embedding and low score using 
resnet-18 embeddings, then we can say that there is possibility of major shifting of the subspace or manifold currently occupied by the data during training. This can help understand possibilities of represenation
shift during training.

### Domain adaptation

After solving the least squares equation, the alpha used can be used for weighing the samples during training. Normally we treat every data point as equally important while training our model. In order to adapt the
model trained on synthetic data to real data, we can weigh the samples by corresponding alpha values during training. The alpha values maybe re-computed mid training like training after every R epochs.

### Coreset selection

We can modify the gradmatch algo used for coreset selection by replacing gradients with difference vectors in order to select a representative coreset.
