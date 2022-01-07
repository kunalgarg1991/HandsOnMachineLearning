"""1. Analysis using .head(), .info(), .describe(), .value_counts()
2. Visualization using histograms, scatter plot, confusion matrix and graph
3. Experimenting with attribute combinations and their effect on above
4. Preparing for machine learning
    1. Splitting traning and testing set, seperating labels and features
    2. seperating numerical and categorical coumns
    3. Transformationol pipeline - import, initialize, fir, tranform,predict, analyse metrics
    4. categorical - ordinal encoder and onehotencoder
    5. Numerical - SimpleImputer, customattributeadder, StandardScalar, MinMaxScalar
    6. Pipelines, columntranfermer
    7. Creating a custom transformer
5. selecting and traning a model - linear regression, randomforest, Decision tree
6. Better evaluation using Crossvalidation - looking at results of various validations
7. Refining the model using gridsearchcv / Randomized search cv
8 Evaluate the system on test set"""

"""1. Analysis using .head(), .info(), .describe(), .value_counts()"""
import pandas as pd
import numpy as np

housing = pd.read_csv("housing.csv")

print (housing.head())
print (housing.info())
print(housing.describe())
print (housing["ocean_proximity"].value_counts())

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
print(housing["income_cat"].value_counts())
housing["income_cat"].hist()

"""2. Visualization using histograms, scatter plot, confusion matrix and graph"""
import matplotlib.pyplot as plt
housing.hist(bins = 50, figsize=(20,15))
plt.savefig("attributes.png", format = "png")
plt.show()

housing["median_income"].hist()
plt.show()

housing.plot(kind="scatter", x="longitude", y="latitude", alpha = 0.1)
plt.show()

housing.plot(kind="scatter", x="longitude", y="latitude", alpha = 0.1, s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend()
plt.show()

"""Looking for correlations"""
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"])

"""from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12.8))
plt.show()"""

"""3. Experimenting with attribute combinations and their effect on above"""

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value",
             alpha=0.2)
plt.axis([0, 5, 0, 520000])
plt.show()
print(housing.describe())


"""4. Preparing for machine learning
    1. Splitting traning and testing set, seperating labels and features
    2. seperating numerical and categorical columns
    3. Transformationol pipeline - import, initialize, fir, tranform,predict, analyse metrics
    4. categorical - ordinal encoder and onehotencoder
    5. Numerical - SimpleImputer, customattributeadder, StandardScalar, MinMaxScalar
    6. Pipelines, columntranfermer
    7. Creating a custom transformer"""

"""Splitting training and test set"""

from sklearn.model_selection import train_test_split
train_set, test_set= train_test_split(housing, test_size=0.2, random_state = 43)
print(test_set.head(5))

"""Seperating the labels and features"""

housing = train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = train_set["median_house_value"].copy()

"""Separating the data into numerical and textual values"""

housing_num = housing.drop("ocean_proximity", axis=1)
housing_cat = housing[["ocean_proximity"]]


"""Preparing the data for machine learning - 

3 transformations on the numerical data - removing null values, adding extra attributes, scaling the values
1 transformation on categorical data - onehotencoder to convert categorical data into matrix of 0 and 1"""

"""Converting categorical data into array
1. Ordinal Encoder - assigns a number to each value of the categorical value starting from 0
2. One hot encoder: converts into sparse matrix - This is better because the ordinal encoder values are 
considered ordered 0<1<!<2,etc"""

from sklearn.preprocessing import OrdinalEncoder
ordinalencoder = OrdinalEncoder()
housing_cat_encoded=ordinalencoder.fit_transform(housing_cat)
print(housing_cat_encoded[:10])
print(ordinalencoder.categories_)

from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder()
housing_cat_oneencoded = onehotencoder.fit_transform(housing_cat)
print(housing_cat_oneencoded)
print(housing_cat_oneencoded.toarray())

"""1. Cleaning the numerical data - removing null values  There are 3 ways of doing it
housing.dropna(subset=["total_bedrooms"])    # option 1 - removing the rows
housing.drop("total_bedrooms", axis=1)       # option 2 - removing the column altogether
median = housing["total_bedrooms"].median()  # option 3 - replacing the blank value with some other value 
housing["total_bedrooms"].fillna(median, inplace=True)

The transformer simpleimputer can be used also for it"""

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing_num)
print(imputer.statistics_)
"""above is also eqaul in values to the median of the columns calculated by the following"""
print(housing_num.median().values)
X=imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index)
print(imputer.strategy)


"""Cleaning the numerical data  - adding custom attributes"""

from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
"""netter to get these values dynamically as
col_names = "total_rooms", "total_bedrooms", "population", "households"
rooms_ix, bedrooms_ix, population_ix, households_ix = [
    housing.columns.get_loc(c) for c in col_names]"""

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

"""Also, housing_extra_attribs is a NumPy array, we've lost the column names (unfortunately, that's a 
problem with Scikit-Learn). To recover a DataFrame, you could run this:"""

housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
print(housing_extra_attribs.head())

"""transformation -3 scaling the numberical data
2 classes can be used to do it - MinMaxScalar and StandardScaler()"""

from sklearn.preprocessing import StandardScaler
stdscalar = StandardScaler()
housing_num_scaled = stdscalar.fit_transform(housing_num)

"""Doing all the transformations for numerical data in one pipeline"""
from sklearn.pipeline import Pipeline
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
housing_num_tr = num_pipeline.fit_transform(housing_num)

"""Doing all transformations for categorical data in one pipeline"""
cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder())
    ])
housing_cat_tr = cat_pipeline.fit_transform(housing_cat)

"""Column transformer can be used to combine all this work"""
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
"""note that housing_prepared is an array"""

"""Selecting and training a model"""
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
housing_predictions = lin_reg.predict(housing_prepared)

from sklearn.metrics import mean_squared_error
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

from sklearn.metrics import mean_absolute_error
lin_mae = mean_absolute_error(housing_labels, housing_predictions)
lin_mae

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


"""Better evaluation using Crpss Validation"""
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

from sklearn.model_selection import cross_val_score
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
pd.Series(lin_rmse_scores).describe()


from sklearn.model_selection import cross_val_score
tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
display_scores(tree_rmse_scores)
pd.Series(tree_rmse_scores).describe()


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
from sklearn.model_selection import cross_val_score
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
pd.Series(forest_rmse_scores).describe()



from sklearn.svm import SVR
svm_reg = SVR(kernel="linear")
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse
svm_scores = cross_val_score(svm_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
svm_rmse_scores = np.sqrt(-svm_scores)
display_scores(svm_rmse_scores)
pd.Series(svm_rmse_scores).describe()

"""Fine tune your model - gridsearchcv"""

from sklearn.model_selection import GridSearchCV
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

"""To return the best params"""
print (grid_search.best_params_)
"""To return the best estimator directly"""
print(grid_search.best_estimator_)

"""Let's look at the score of each hyperparameter combination tested during the grid search:"""
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
pd.DataFrame(grid_search.cv_results_)

"""Randomized Search CV to control the number of iterations to choose the best estimator """
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
    'n_estimators': randint(low=1, high=200),
    'max_features': randint(low=1, high=8),
}

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
"""n_iter controls the number of iterations"""
rnd_search.fit(housing_prepared, housing_labels)

cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


"""Analyze the Best Models and Their Errors"""
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
# cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
"""Drop the features with less importance"""


"""Evaluate teh system on test set"""
final_model = grid_search.best_estimator_

X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)


"""Full pipeline with prep and prediction"""

full_pipeline_with_predictor = Pipeline([
        ("preparation", full_pipeline),
        ("linear", LinearRegression())
    ])

full_pipeline_with_predictor.fit(housing, housing_labels)
full_pipeline_with_predictor.predict(some_data)

 """Model persistance using joblib"""

my_model = full_pipeline_with_predictor
import joblib
joblib.dump(my_model, "my_model.pkl") # DIFF
#...
my_model_loaded = joblib.load("my_model.pkl") # DIFF