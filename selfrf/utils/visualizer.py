from typing import Callable, Iterable, List, Optional, Tuple

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

import torch


class ViewInvarianceVisualizer():
    """Visualizes self-supervised learning data augmentation views.

    This visualizer shows the original signal and its transformed view side by side,
    helping to understand and debug SSL data augmentation pipelines. It supports
    both spectrogram and IQ data visualization.

    Args:
        data_loader (DataLoader): PyTorch DataLoader providing signal samples
        ssl_transform (Transform): SSL transformation pipeline (e.g., BYOLTransform)
        visualize_transform (callable, optional): Transform to convert data for plotting
        num_samples (int, optional): Number of samples to visualize. Defaults to 8

    Example:
        >>> visualizer = SSLViewVisualizer(
        ...     data_loader=train_loader,
        ...     ssl_transform=BYOLTransform(),
        ...     visualize_transform=complex_spectrogram_to_magnitude
        ... )
        >>> for fig in visualizer:
        ...     plt.show()
    """

    def __init__(
        self,
        data_loader,
        visualize_transform: Optional[Callable] = None,
    ):

        self.data_loader = iter(data_loader)
        self.visualize_transform = visualize_transform

    def __iter__(self) -> Iterable:
        self.data_iter = iter(self.data_loader)
        return self  # type: ignore

    def __next__(self) -> Figure:
        batch = next(self.data_iter)

        return self._visualize(batch)

    def _visualize(self, batch: Tuple[List[torch.Tensor], torch.Tensor, List[str]]) -> Figure:
        """Create side-by-side visualization of view pairs."""

        batch_size = len(batch)
        views = batch[0]
        views1, views2 = views[0], views[1]

        # Create figure with subplots for each sample
        nrows = batch_size
        fig, axes = plt.subplots(nrows, 2, figsize=(
            10, 5*batch_size), frameon=True)

        # Apply visualization transform in batches
        if self.visualize_transform:
            views1 = self.visualize_transform(views1)
            views2 = self.visualize_transform(views2)

        # Loop through batch
        for i in range(batch_size):
            # Get views for current sample

            view1 = views1[i]
            view2 = views2[i]

            # Ensure views are in correct format
            if isinstance(view1, torch.Tensor):
                view1 = view1.numpy()
            if isinstance(view2, torch.Tensor):
                view2 = view2.numpy()

            # ensure views are 2D
            if view1.ndim == 3:
                view1 = view1.squeeze(0)
            if view2.ndim == 3:
                view2 = view2.squeeze(0)

            # Plot views
            axes[i, 0].imshow(view1, aspect='auto', cmap='jet')
            axes[i, 1].imshow(view2, aspect='auto', cmap='jet')

            # Remove ticks
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])

        # Add column titles
        axes[0, 0].set_title('View 1')
        axes[0, 1].set_title('View 2')

        plt.tight_layout()
        return fig
