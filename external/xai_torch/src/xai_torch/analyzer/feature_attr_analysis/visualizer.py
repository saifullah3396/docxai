from captum.attr import visualization as viz
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from xai_torch.utilities.general import get_matplotlib_grid


class ImageVisualizer:
    default_cmap = LinearSegmentedColormap.from_list(
        "custom blue", [(0, "#ffffff"), (0.25, "#000000"), (1, "#000000")], N=256
    )

    @classmethod
    def visualize_image_attr(cls, attribution_map, image, fig, ax, title=None):
        try:
            return viz.visualize_image_attr(
                attribution_map,
                original_image=image,
                method="heat_map",
                sign="all",
                plt_fig_axis=(fig, ax),
                title=title,
                use_pyplot=False,
            )
        except AssertionError as e:
            pass

    @classmethod
    def visualize_image_attr_grid(cls, attr_maps, images, show=False):
        rows = len(images)
        cols = 2
        fig, gs = get_matplotlib_grid(rows, cols)

        # create rows for grid
        for idx, (image, attr_map) in enumerate(zip(images, attr_maps)):
            ax = plt.subplot(gs[idx, 0])
            ax.imshow(image.permute(1, 2, 0).numpy())
            ax.set_xticks([])
            ax.set_yticks([])

            ax = plt.subplot(gs[idx, 1])
            cls.visualize_image_attr(attr_map.permute(1, 2, 0).numpy(), image.permute(1, 2, 0).numpy(), fig, ax)

        return fig
