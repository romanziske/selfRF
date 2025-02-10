from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import List
import numpy as np

class VisualizeTSNE():

    def __init__(self, x: List[np.ndarray], y: List[str], class_list: List[str],) -> None:
        self.x = x
        self.y = y
        self.class_list = class_list

        # Create a color dictionary for each unique label
        colors = plt.cm.rainbow(np.linspace(0, 1, len(class_list)))
        self.color_dict = dict(zip(class_list, colors))

    def visualize(self, save_path=None):
        # Create figure and axis
        _, ax = plt.subplots(figsize=(16, 9))

        # Run TSNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(self.x)

        # Plot each class
        for label, color in self.color_dict.items():
            indices = np.nonzero(np.array(self.y) == label)
            # Select the corresponding t-SNE results and plot them
            plt.scatter(tsne_results[indices, 0], tsne_results[indices,
                        1], color=color, label=label, alpha=0.5)

        # Customize plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return plt.gcf()
