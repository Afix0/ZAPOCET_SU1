# Mushroom species clustering and edibility classification

## **Overview**
This project applies **Agglomerative Clustering**, a hierarchical unsupervised machine learning technique, to group sentences based on their semantic similarity. The sentences are first transformed into vector representations using the **Word2Vec** model, and then clusters are formed based on **cosine similarity** between these embeddings. The project demonstrates how hierarchical clustering can be utilized for natural language processing tasks.


---

## **Project Goals**

### **1. Mushroom Species Clustering**
- Use hierarchical clustering to group mushroom species based on their features and characteristics.
- Generate embeddings for the data and apply clustering techniques to reveal natural groupings.
- Visualize the clusters to understand species similarities and relationships.

### **2. Decision Tree for Edibility Classification**
- Build a decision tree model to classify mushrooms as edible or poisonous based on their features.
- Evaluate the decision tree's accuracy and interpret its classification rules.
- Explore how feature importance impacts classification outcomes.

---


## **Dependencies**
The following libraries are required to run the project:

- `numpy` (for numerical computations)
- `gensim` (for Word2Vec embeddings)
- `matplotlib` (for visualization)
- `pandas` (for data manipulation)
- `collections` (for Counter functionality)
- `scikit-learn` (for clustering and preprocessing)
- `ucimlrepo` (to fetch UCI datasets)


## **Files in the Repository**
- **mushroom_clustering.ipynb**: Contains the main implementation of Agglomerative Clustering for text data.
- **mushroom_classification.ipynb**: Contains decision tree classifier 
- **SU1_zapocet_functions.py**: Helper functions for sentence embedding calculation and visualization.
