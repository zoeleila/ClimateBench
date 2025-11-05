import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs

class EvaluationPlots():
    """Class for plotting evaluation metrics for 2D times series data.
    Data is assumed to be in the shape (time, lat, lon) or (time, y, x)."""
    def __init__(self,
                 simulation_name=None,
                 var_name=None,
                 domain=None,
                 projection=ccrs.PlateCarree()):
        pass
    def plot_time_series(self, y_true, y_pred, save_path=None):
        """Plot time series of true vs predicted values."""
        _, ax = plt.subplots(figsize=(12,6))
        time = np.arange(y_true.shape[0])
        ax.plot(time, y_true, label='True', color='blue')
        ax.plot(time, y_pred, label='Predicted', color='orange')
        ax.set_xlabel('Time')
        ax.set_ylabel(self.var_name if self.var_name else 'Value')
        ax.legend()
        ax.set_title('Time Series of True vs Predicted Values')
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_spatial_map(self, 
                         y_true, 
                         y_pred, 
                         time_index=None, 
                         save_path=None,
                         cmap='viridis'):
        """Plot spatial map of true vs predicted values at a specific time index."""
        if time_index is None:
            y_true = np.mean(y_true, axis=0)
            y_pred = np.mean(y_pred, axis=0)
        else:
            y_true = y_true[time_index]
            y_pred = y_pred[time_index]
        fig, axes = plt.subplots(1, 2, figsize=(14,6), subplot_kw={'projection': self.projection})
        im1 = axes[0].imshow(y_true, 
                             origin='lower', 
                             transform=self.projection, 
                             cmap=cmap)
        axes[0].set_title('True Values')
        plt.colorbar(im1, ax=axes[0], orientation='vertical')
        im2 = axes[1].imshow(y_pred, 
                             origin='lower', 
                             transform=self.projection, 
                             cmap=cmap)
        axes[1].set_title('Predicted Values')
        plt.colorbar(im2, ax=axes[1], orientation='vertical')
        fig.suptitle(f'Spatial Map at time index {time_index}' if time_index is not None else 'Mean Spatial Map')
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


def plot_histogram(data, stats: dict, title: str=None, save_dir:str=None, xlim_dict: dict = None):
    """
    Affiche un subplot 2x2 avec les histogrammes des 4 variables.
    
    Args:
        data: np.ndarray de forme (..., 4)
        stats: dict contenant mean, std, min, max pour chaque variable
        title: titre global de la figure
        save_dir: chemin du fichier de sortie (ex: './figures/hist.png')
    """
    
    channels = list(stats.keys())
    n_var = len(channels)
    
    n_cols = int(np.ceil(np.sqrt(n_var)))
    n_rows = int(np.ceil(n_var / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    for i, channel in enumerate(channels):
        array = data[..., i].flatten()
        mean = stats[channel]['mean']
        print(mean)
        std = stats[channel]['std']
        print(std)
        vmin = stats[channel]['min']
        vmax = stats[channel]['max']

        # Histogramme
        hist, edges = np.histogram(array, bins=50, range=(vmin, vmax), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        
        ax = axes[i]
        ax.bar(centers, hist, align='center', width=np.diff(edges), alpha=0.5, color='blue', label='Density')
        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'$\mu$ = {mean:.2f}')
        
        if xlim_dict and channel in xlim_dict and xlim_dict[channel] is not None:
            ax.set_xlim(tuple(xlim_dict[channel]))
        #ax.set_xscale("log")
        ax.set_xlabel(channel, fontsize=16)
        ax.set_ylabel('Density', fontsize=12)
        ax.grid(True)
        ax.legend(fontsize=10, loc='upper right')
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_dir:
        plt.savefig(save_dir)
        plt.close()
    else:
        plt.show()