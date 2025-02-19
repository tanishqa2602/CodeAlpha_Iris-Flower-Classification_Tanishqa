# Iris Flower Classification using K-Nearest Neighbors (KNN)

## Project Overview
This project implements a K-Nearest Neighbors (KNN) classifier to classify iris flowers into three species: Setosa, Versicolor, and Virginica. The dataset consists of four features: sepal length, sepal width, petal length, and petal width. Principal Component Analysis (PCA) is also applied for dimensionality reduction and visualization.

## Features
- Data preprocessing and loading
- Exploratory Data Analysis (EDA) with histograms and pairwise plots
- Standardization of features
- Training a K-Nearest Neighbors (KNN) classifier
- Model evaluation with accuracy score and confusion matrix
- Decision boundary visualization using PCA

## Dataset
The dataset contains the following columns:
- `sepal_length`: Length of the sepal (cm)
- `sepal_width`: Width of the sepal (cm)
- `petal_length`: Length of the petal (cm)
- `petal_width`: Width of the petal (cm)
- `species`: The species of the flower (`setosa`, `versicolor`, `virginica`)

## Requirements
To run this project, install the required dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Running the Analysis
1. Clone the repository:
```bash
git clone https://github.com/yourusername/iris-classification.git
```
2. Navigate to the project folder:
```bash
cd iris-classification
```
3. Run the Python script:
```bash
python 1.1_code_alpha.py
```

## Visualizations
- Histograms of feature distributions
- Pairwise feature relationships
- Confusion matrix for model evaluation
- Decision boundaries using PCA

## Results
- The KNN classifier achieves a competitive accuracy in classifying iris species.
- The confusion matrix highlights correct and incorrect classifications.
- PCA visualization helps understand decision boundaries in reduced dimensions.

## Contribution
Feel free to fork this repository, report issues, or submit pull requests to enhance the project.

## License
This project is licensed under the MIT License.

---
Created by [Your Name]
