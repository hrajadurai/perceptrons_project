# __Perceptron-Based Classification Model__

The perceptron-based classification model is a linear binary classifier that uses a single-layer neural network to make predictions based on weighted inputs and a threshold activation function.

Let's understand how to build a perceptron-based classification model.

## Steps to be followed:
1. Import the required libraries
2. Read a CSV file
3. Display the data
4. Perform data preprocessing and splitting
5. Fit the model
6. Predict the model

### Step 1: Import the required libraries

- Import necessary modules for numerical computations and defines functions for exponential calculations, array operations, random number generation, and matrix multiplication.

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
%matplotlib inline

### Step 2: Read a CSV file

- Load the data from a CSV file.
- Read a CSV file using the **pd.read_csv()** function.


data = pd.read_csv("mnist_train.csv")

### Step 3: Display the data

- The __head()__ is used to retrieve the few rows of the dataset named __mnist_train__.

data.head()

**Observation:**

- As a result, the display consists of **5** rows and **785** columns.

### Step 4: Perform data preprocessing and splitting

- To check if there are any missing values in the dataset, use the **isnull()** function combined with the **any()** function in pandas.
- The **data.iloc[:,1:]** selects all columns except the first one from the DataFrame data and assigns them to **df_x**, and __data.iloc[:,0]__ selects only the first column and assigns it to **df_y**.
- Perform a train-test split on the input data **df_x** and **df_y**, allocating **80%** of the data for training **x_train** and **y_train** and **20%** for testing **x_test** and **y_test**, with a random state of **4** for reproducibility.
- Create an instance of the perceptron classifier **per**.
- Initialize the classifier with default parameters, allowing it to be used for classification tasks.


if data.isnull().values.any():
    data = data.fillna(0)

df_x = data.iloc[:,1:]
df_y = data.iloc[:,0]

x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.2, random_state=4)

per = Perceptron()

### Step 5: Fit the model

- Fit the perceptron-based classification model for the **x_train** and **y_train** datasets to supervised learning for various binary classifiers by defining perceptrons.


per.fit(x_train, y_train)


### Step 6: Predict the model

- Predict the model for **x_test**.

pred = per.predict(x_test)

pred

**Observations:**

- The prediction of **X_test** is presented above in an array format which represents a sequence of values.
- Each value in the array corresponds to a specific element within a dataset or sequence. However, without additional context, the exact meaning or source of these values within the array cannot be precisely determined.

**Check the Accuracy Score**
- Import the **accuracy_score** function from the **sklearn.metrics** module. This function is used to compute the accuracy of a classifier's predictions.
- **y_test** is the actual labels for the test set. These are the 'true' values that the model is trying to predict.
- **pred** is the predicted labels for the test set, as predicted by the model.

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, pred)

print("Accuracy of the model: ", accuracy)

**Observation:**
- The **accuracy_score** function compares these two arrays and returns the proportion of correct predictions, which is the accuracy of the model. This accuracy score is then stored in the variable accuracy.
