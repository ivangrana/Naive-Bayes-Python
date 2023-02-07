import pandas as pd
import numpy as np

# Load data into a pandas dataframe
df = pd.read_csv('data.csv')

# Split the data into training and test sets
train_data = df.sample(frac=0.8, random_state=200)
test_data = df.drop(train_data.index)

# Compute class probabilities
class_probs = train_data['class'].value_counts(normalize=True)

# Compute the probability of each feature given the class
feature_probs = {}
for c in class_probs.index:
    feature_probs[c] = {}
    class_data = train_data[train_data['class'] == c]
    for feature in df.columns[:-1]:
        feature_probs[c][feature] = class_data[feature].value_counts(normalize=True)

# Define a function to make predictions
def predict(data, class_probs, feature_probs):
    predictions = []
    for i, row in data.iterrows():
        probs = {}
        for c in class_probs.index:
            prob = class_probs[c]
            for feature in df.columns[:-1]:
                if row[feature] in feature_probs[c][feature]:
                    prob *= feature_probs[c][feature][row[feature]]
                else:
                    prob = 0
            probs[c] = prob
        predictions.append(max(probs, key=probs.get))
    return predictions

# Make predictions on the test data
predictions = predict(test_data, class_probs, feature_probs)

# Evaluate the accuracy of the model
accuracy = np.mean(predictions == test_data['class'].tolist())
print('Accuracy:', accuracy)
