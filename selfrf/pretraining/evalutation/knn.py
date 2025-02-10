from typing import List
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class EvaluateKNN():

    def __init__(self, x: List[np.ndarray], y: List[str], split: float = 0.8, shuffel: bool = True, n_neighbors: int = 50):
        # Split the data

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, train_size=split, shuffle=shuffel, random_state=42)
        self.n_neighbors = n_neighbors

    def evaluate(self) -> float:
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)

        # Fit the model
        knn.fit(self.x_train, self.y_train)

        # Predict the labels
        pred_labels = knn.predict(self.x_test)

        # Calculate accuracy
        acc = accuracy_score(self.y_test, pred_labels)
        return acc
