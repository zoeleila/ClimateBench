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
