import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def plot_decision_boundary(clf, X, y, title):
 """
 Helper function to visualize the decision boundary of an SVM classifier.
 Creates a grid of points, predicts the class for each, and plots the contour.
 """
 plt.figure(figsize=(6, 5))
 ax = plt.gca()

 # Create a mesh grid with a small step size (0.02) for smooth contours
 x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
 y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
 xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
 np.arange(y_min, y_max, 0.02))

 # Predict the classification for every point in the mesh grid
 Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
 Z = Z.reshape(xx.shape)

 # Plot the filled contours and the data points
 ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)
 ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=25)

 ax.set_title(title)
 ax.set_xticks(())
 ax.set_yticks(())
 plt.show()

# Generate Non-Linear Data (Concentric Circles)
X, y = make_circles(n_samples=300, factor=0.3, noise=0.1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply Standard Scaling! (Critical for SVM Distance Calculations)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Train a Linear SVM (Will perform poorly on circular data)
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train_scaled, y_train)

linear_preds = svm_linear.predict(X_test_scaled)
linear_acc = accuracy_score(y_test, linear_preds)

print(f"Linear Kernel Accuracy: {linear_acc:.2f}")

plot_decision_boundary(svm_linear, X_train_scaled, y_train,
 f"Linear SVM (Accuracy: {linear_acc:.2f})\nFailed to capture non-linearity")

# Train a Polynomial SVM (degree controls the polynomial power)
svm_poly = SVC(kernel='poly', degree=3, C=1.0)
svm_poly.fit(X_train_scaled, y_train)

poly_preds = svm_poly.predict(X_test_scaled)
poly_acc = accuracy_score(y_test, poly_preds)

print(f"Polynomial Kernel Accuracy: {poly_acc:.2f}")

plot_decision_boundary(svm_poly, X_train_scaled, y_train,
 f"Polynomial SVM (Accuracy: {poly_acc:.2f})\nModerate non-linear fit")


# Train an RBF (Non-Linear) SVM
# gamma controls the 'spread' of the kernel, C controls regularization
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train_scaled, y_train)

rbf_preds = svm_rbf.predict(X_test_scaled)
rbf_acc = accuracy_score(y_test, rbf_preds)

print(f"RBF Kernel Accuracy: {rbf_acc:.2f}")

plot_decision_boundary(svm_rbf, X_train_scaled, y_train,
 f"RBF Kernel SVM (Accuracy: {rbf_acc:.2f})\nSuccessfully separated using RBF")
