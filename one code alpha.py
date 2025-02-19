import pandas as pd

# Create a dataset manually (the Iris dataset is typically 150 samples with 4 features)
data = {
    "sepal_length": [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9, 5.4, 4.8, 4.8, 4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1],
    "sepal_width": [3.5, 3.0, 3.2, 3.4, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.7, 3.4, 3.4, 3.0, 4.0, 4.1, 3.9, 3.5, 3.8, 3.8],
    "petal_length": [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.3, 1.5, 1.7, 1.6, 1.5, 1.6, 1.2, 1.3, 1.4, 1.7, 1.7, 1.5],
    "petal_width": [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.3, 0.3],
    "species": ["setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", 
                "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor"]
}

# Create a DataFrame
df = pd.DataFrame(data)

print(df)

#to do df to csv

df.to_csv('iris_flower_data.csv', index=False)
print("iris_flower_data.csv")

data["species"] += ["virginica"] * 10
data["sepal_length"] += [6.2, 6.3, 6.1, 6.4, 6.5, 7.1, 6.6, 6.8, 6.3, 6.1]
data["sepal_width"] += [3.4, 2.9, 3.0, 2.8, 3.0, 3.1, 3.0, 3.1, 3.0, 2.9]
data["petal_length"] += [5.4, 5.6, 5.1, 5.3, 5.8, 5.9, 5.6, 5.8, 5.7, 5.4]
data["petal_width"] += [2.3, 2.4, 2.3, 2.3, 2.2, 2.3, 2.3, 2.3, 2.3, 2.2]

# Now update the DataFrame with the extended data
df = pd.DataFrame(data)
df.to_csv('iris_flower_data.csv', index=False)

print("Updated Dataset saved to iris_flower_data.csv")


