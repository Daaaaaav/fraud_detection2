import numpy as np

# Assuming y_train and y_test are loaded as numpy arrays
y_train = np.load('processed_data.npz')['y_train']
y_test = np.load('processed_data.npz')['y_test']

# Get the counts of each class in y_train and y_test
train_class_counts = np.bincount(y_train)
test_class_counts = np.bincount(y_test)

print(f"Class distribution in y_train: {train_class_counts}")
print(f"Class distribution in y_test: {test_class_counts}")

# Optionally, check the percentage of each class
train_percentage = train_class_counts / len(y_train) * 100
test_percentage = test_class_counts / len(y_test) * 100

print(f"Class percentages in y_train: {train_percentage}")
print(f"Class percentages in y_test: {test_percentage}")
