# Analyzing-Student-Data

# Predicting Student Admissions with Neural Networks

This project focuses on predicting student admissions to graduate school at UCLA using a neural network. The prediction is based on three key factors:

- GRE Scores (Test)
- GPA Scores (Grades)
- Class rank (1-4)

The dataset used for this project can be found [here](http://www.ats.ucla.edu/).

## Table of Contents
- [Loading the Data](#loading-the-data)
- [Plotting the Data](#plotting-the-data)
- [One-Hot Encoding the Rank](#one-hot-encoding-the-rank)
- [Scaling the Data](#scaling-the-data)
- [Splitting the Data into Training and Testing Sets](#splitting-the-data-into-training-and-testing)
- [Splitting the Data into Features and Targets](#splitting-the-data-into-features-and-targets)
- [Training the 2-layer Neural Network](#training-the-2-layer-neural-network)
- [Calculating Accuracy on the Test Data](#calculating-accuracy-on-the-test-data)

## Loading the Data

To load and format the data, we use Pandas and Numpy. You can read more about these libraries in their documentation:

- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Numpy Documentation](https://docs.scipy.org/)

```python
import pandas as pd
import numpy as np

# Reading the CSV file into a Pandas DataFrame
data = pd.read_csv('student_data.csv')

# Printing out the first 10 rows of our data
data[:10]
