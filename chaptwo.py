import pandas as pd
housing = pd.read_csv("housing.csv")
print (housing.head())
print (housing.info())
print (housing["ocean_proximity"].value_counts())
print(housing.describe())

import matplotlib.pyplot as plt
housing.hist(bins = 50, figsize=(20,15))
plt.savefig("attributes.png", format = "png")
plt.show()

import numpy as np
from sklearn.model_selection import train_test_split
train_Set, test_set= train_test_split(housing, test_size=0.2, random_state = 43)
print(test_set.head(5))

housing["median_income"].hist()
plt.show()

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

print(housing["income_cat"].value_counts())

housing["income_cat"].hist()
