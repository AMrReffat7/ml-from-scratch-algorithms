# ML From Scratch Algorithms

A collection of classic machine learning algorithms implemented from scratch in Python using only NumPy.

## Included Algorithms

- Linear Regression (+ L1 and L2 Regularization)
- K Nearest Neighbors
- Gaussian Naive Bayes
- Random Forest (entropy)
- KMeans Clustering
- DBSCAN Clustering

## Usage

Clone the repo:

```bash
git clone https://github.com/AMrReffat7/ml-from-scratch-algorithms.git
````

Install NumPy:

```bash
pip install numpy
```

Example (KMeans):

```python
from kmeans import KMeansClustering

model = KMeansClustering(n_clusters=3)
model.fit(data)
labels = model.predict(data)
```

## ✨ About
This project is my personal favorite and was created to practice and explore each algorithm in depth, so I can understand them thoroughly beyond just using libraries like scikit-learn.
