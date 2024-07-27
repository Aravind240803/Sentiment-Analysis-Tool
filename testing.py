import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Example actual and predicted values
y_true = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1]  # Actual labels
y_pred = [0, 0, 0, 1, 0, 1, 1, 0, 1, 1]  # Predicted labels

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Print the confusion matrix with an explanation
print("Confusion Matrix:")
print(cm)
print("\nExplanation:")
print("The confusion matrix is a table that is often used to describe the performance of a classification model.")
print("Each row of the matrix represents the instances in an actual class, while each column represents the instances in a predicted class.")
print("Here's what each value means:")
print(f"True Negative (TN): {cm[0, 0]} - The model correctly predicted negative cases.")
print(f"False Positive (FP): {cm[0, 1]} - The model incorrectly predicted positive cases.")
print(f"False Negative (FN): {cm[1, 0]} - The model incorrectly predicted negative cases.")
print(f"True Positive (TP): {cm[1, 1]} - The model correctly predicted positive cases.")

# Visualize the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix Visualization")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
