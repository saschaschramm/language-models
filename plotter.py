import matplotlib.pyplot as plt
import numpy as np
import matplotlib


class Plotter:
    def plot_relative_frequency(
        self,
        labels,
        feature_0_space,
        feature_1_space,
        probs,
        filename,
        x_name,
        y_name,
        mask=False,
    ):
        probs = np.asarray(probs)
        probs = probs.reshape((feature_0_space.shape[0], feature_1_space.shape[0]))
        labels = np.asarray(labels)
        labels = labels.reshape((feature_0_space.shape[0], feature_1_space.shape[0]))

        plt.style.use("dark_background")

        cmap = matplotlib.cm.get_cmap("binary_r")

        plt.pcolormesh(
            feature_0_space,
            feature_1_space,
            probs,
            shading="auto",
            vmin=probs.min(),
            vmax=probs.max(),
        )

        if mask:
            plt.pcolormesh(
                feature_0_space,
                feature_1_space,
                labels,
                shading="auto",
                cmap=cmap,
                vmin=labels.min(),
                vmax=labels.max(),
            )
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.savefig(filename, dpi=300)
