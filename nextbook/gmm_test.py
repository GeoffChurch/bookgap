import time
import torch
import torch.nn as nn
import numpy as np
import math
import plotly.graph_objects as go
from scipy.stats import multivariate_normal # Keep for plotting if needed
import itertools # Added for looping through combinations
from tqdm.auto import tqdm # Import tqdm

def optimize_kde_bandwidths(points: np.ndarray, power: float = 1.0, include_self: bool = False, n_epochs: int = 100, lr: float = 0.01, initial_log_sigma: float = 0.0, min_log_sigma: float = -20.0, show_progress: bool = False):
    """
    Optimizes per-point bandwidths (sigmas) for a Gaussian KDE using a modified
    objective based on the sum of powered densities. Allows including the point's
    own kernel contribution in its score calculation.

    Objective: Maximize sum_i log( sum_{j in J_i} N(x_i | x_j, sigma_j^2)^power )
        where J_i = {1, ..., n} if include_self is True
        and   J_i = {1, ..., n} \ {i} if include_self is False (Leave-One-Out)

    Args:
        points: A numpy array of shape (n_points, n_dims) containing the data.
        power: The power to which individual kernel densities N(x_i|x_j, sigma_j^2)
               are raised before summing for the score of x_i.
        include_self: If True, include the contribution of N(x_i | x_i, sigma_i^2)
                      when calculating the score for point x_i. If False, use
                      the standard Leave-One-Out approach.
        n_epochs: Number of optimization iterations.
        lr: Learning rate for the Adam optimizer.
        initial_log_sigma: Initial value for the logarithm of sigma for all points.
        min_log_sigma: Minimum allowed value for log_sigma to prevent collapse.
        show_progress: If True, display a tqdm progress bar during optimization.

    Returns:
        A numpy array of shape (n_points,) containing the optimized sigma
        values for each point's kernel.
    """
    n_points, n_dims = points.shape
    if not include_self and n_points <= 1:
        raise ValueError("Need at least 2 points for Leave-One-Out KDE (include_self=False).")
    if n_points == 0:
        return np.array([], dtype=np.float32)

    # Convert points to tensor
    X = torch.tensor(points, dtype=torch.float32)
    device = X.device # Get device for tensor creation later

    # Initialize log_sigmas as learnable parameters
    log_sigmas = nn.Parameter(torch.full((n_points,), initial_log_sigma, dtype=torch.float32, device=device))

    optimizer = torch.optim.Adam([log_sigmas], lr=lr)

    # Precompute constants
    log_2pi = math.log(2 * math.pi)

    # Wrap the loop with tqdm
    with tqdm(range(n_epochs), desc="Optimizing Bandwidths", disable=not show_progress, leave=False) as epoch_iterator:
        for epoch in epoch_iterator:
            optimizer.zero_grad()

            # Ensure sigmas don't get too small
            with torch.no_grad():
                log_sigmas.clamp_(min=min_log_sigma)

            # --- Calculate Log Gaussian Kernels ---
            sigmas_j = torch.exp(log_sigmas)
            variances_j = sigmas_j.pow(2) # Shape (n_points,)

            diff = X.unsqueeze(1) - X.unsqueeze(0) # Shape (n_points, n_points, n_dims)
            sq_dists = torch.sum(diff**2, dim=2)    # Shape (n_points, n_points)

            variances_j_broadcast = variances_j.unsqueeze(0) # Shape (1, n_points)

            # Avoid log(0) if variance is clamped to near zero
            safe_variances_j_broadcast = torch.clamp(variances_j_broadcast, min=1e-20)
            log_norm_const = -0.5 * n_dims * (log_2pi + torch.log(safe_variances_j_broadcast))
            # Avoid division by zero if variance is clamped to near zero
            log_exponent = -0.5 * sq_dists / safe_variances_j_broadcast

            # log_kernels[i, j] = log N(X_i | X_j, sigma_j^2)
            log_kernels = log_norm_const + log_exponent # Shape (n_points, n_points)

            # --- Calculate Modified Scores ---
            # We want log( sum_{j in J_i} N(X_i | X_j, sigma_j^2)^power )
            # This is log( sum_{j in J_i} exp( power * log N(X_i | X_j, sigma_j^2) ) )

            # Calculate log of powered kernels: power * log N_ij
            log_kernels_powered = log_kernels * power

            if include_self:
                # Use all kernels (including j=i)
                log_kernels_target = log_kernels_powered
            else:
                # Use logsumexp for stability. Exclude the diagonal (j=i) term.
                log_kernels_target = log_kernels_powered.clone()
                # Set log(0) for j=i by setting to -inf
                log_kernels_target.fill_diagonal_(-float('inf'))

            # log_sum_exp_target_kernels[i] = log( sum_{j in J_i} exp(log_kernels_powered[i, j]) )
            #                                = log( sum_{j in J_i} N_ij^power )
            log_sum_exp_target_kernels = torch.logsumexp(log_kernels_target, dim=1) # Shape (n_points,)

            # log_s_i is the log of the score for point i
            log_s_i = log_sum_exp_target_kernels # Shape (n_points,)

            # --- Objective Function ---
            # Maximize: sum_i log( s_i ) = sum_i log_s_i
            # Minimize: -sum_i log_s_i

            # Handle potential -inf if a point is very isolated or power is large
            finite_log_s = torch.isfinite(log_s_i)
            if not finite_log_s.all():
                # print(f"Warning: Non-finite log-scores detected at epoch {epoch}. Clamping may occur.")
                # Replace -inf with a very small number
                log_s_i = torch.where(finite_log_s, log_s_i, torch.tensor(-1e10, device=device))

            loss = -torch.sum(log_s_i) # Minimize the negative sum of log scores

            # --- Backpropagation and Optimization Step ---
            if torch.isnan(loss):
                print(f"Error: Loss is NaN at epoch {epoch}. Stopping optimization.")
                break

            loss.backward()
            optimizer.step()

            # Update tqdm description with loss (optional)
            epoch_iterator.set_postfix(loss=loss.item())

    # Return the optimized standard deviations (sigmas)
    final_sigmas = torch.exp(log_sigmas.detach()).cpu().numpy()
    return final_sigmas

# Example Usage:
if __name__ == '__main__':
    # ... (previous imports) ...
    import plotly.graph_objects as go
    import itertools
    from tqdm.auto import tqdm # Make sure tqdm is imported here too if not already

    # Generate some sample data (e.g., two clusters + one isolated point)
    np.random.seed(42)
    points1 = np.random.randn(30, 2) * 0.5 + np.array([0, 0])
    points2 = np.random.randn(30, 2) * 0.5 + np.array([3, 3])
    points3 = np.array([[0, 4]]) # Isolated point
    points = np.vstack([points1, points2, points3])
    n_points, n_dims = points.shape

    if n_dims != 2:
        print("Plotting example currently only supports 2D data.")
        exit()

    # --- Define parameter ranges ---
    power_values = [1.0, 2.0]
    include_self_options = [False]
    n_epochs_val = 2000
    lr_val = 0.05
    init_log_sigma_val = -0.5
    min_log_sigma_val = -4.0
    k_neighborss = [1,2]

    # --- Loop through combinations ---
    for power_value, include_self_option, k_neighbors in itertools.product(power_values, include_self_options, k_neighborss):
        print("-" * 40)
        print(f"Running optimization for: power={power_value}, include_self={include_self_option}")

        start_time = time.time()
        # Optimize bandwidths using show_progress
        optimized_sigmas = optimize_kde_bandwidths(
            points,
            power=power_value,
            include_self=include_self_option, # Pass the option
            n_epochs=n_epochs_val,
            lr=lr_val,
            initial_log_sigma=init_log_sigma_val,
            min_log_sigma=min_log_sigma_val,
            show_progress=True # Use show_progress instead of verbose
        )
        if (diff := time.time() - start_time) > 1:
            print(f"Optimization took {diff:.2f} seconds.")
        # ... (rest of the print statements if needed) ...

        # print("Generating plot...")
        start_time = time.time()

        # 1. Create a grid for the surface plot
        x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
        y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1
        grid_res = 500 # Resolution of the grid
        x_grid = np.linspace(x_min, x_max, grid_res)
        y_grid = np.linspace(y_min, y_max, grid_res)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.vstack([xx.ravel(), yy.ravel()]).T # Shape (grid_res*grid_res, 2)

        # 2. Calculate KDE density on the grid using optimized sigmas
        kde_density = np.zeros(grid_points.shape[0])
        variances = optimized_sigmas**2
        for j in range(n_points):
            # Manual calculation (faster for isotropic)
            sq_dist = np.sum((grid_points - points[j])**2, axis=1)
            var_j = variances[j]
            # Handle potential zero variance if min_log_sigma is too low or optimization fails
            if var_j <= 1e-10: continue # Skip points with near-zero variance
            norm_const = 1.0 / (2 * math.pi * var_j) # For 2D
            exponent = -0.5 * sq_dist / var_j
            pdf_values = norm_const * np.exp(exponent)
            kde_density += pdf_values

        # Normalize density (average contribution) - helps keep z-axis scale consistent
        if n_points > 0:
            kde_density /= n_points
        zz = kde_density.reshape(xx.shape)

        # Calculate log density, adding a small epsilon to avoid log(0)
        epsilon = 1e-10
        log_zz = np.log(np.maximum(zz, epsilon)) # Use maximum to avoid log(<0) if epsilon is negative somehow
        log_zz_min = log_zz.min()
        log_zz_max = log_zz.max()
        log_zz_range = log_zz_max - log_zz_min

        # 3. Create Plotly figure
        fig = go.Figure()

        # Add the KDE contour plot using log density for coloring
        fig.add_trace(go.Contour(
            z=log_zz, # Use log density for coloring and contour levels
            x=x_grid,
            y=y_grid,
            colorscale='Viridis', # Choose a colorscale
            showscale=True, # Show colorbar for contours
            colorbar=dict(title='Log Density'), # Add title to colorbar
            contours=dict(
                coloring='heatmap', # Color contours based on z-value (log density)
                # showlabels = True, # Show contour level labels
                labelfont = dict( # Label font properties
                    size = 12,
                    color = 'white',
                ),
                # Adjust start/end/size based on log density range
                start=log_zz_min + log_zz_range * 0.05,
                end=log_zz_max - log_zz_range * 0.05,
                size=log_zz_range / 15 if log_zz_range > 0 else 0 # Avoid division by zero
            ),
            name='Log Density Contours' # Update trace name
        ))

        # Add the original data points as a scatter plot
        fig.add_trace(go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode='markers',
            marker=dict(
                size=6,
                color='red', # Point color
                opacity=0.9,
                symbol='x' # Use 'x' marker for better visibility on contours
            ),
            name='Data Points'
        ))

        # --- Add Nearest Neighbor Arrows ---
        if n_points > 1 and k_neighbors > 0:
            # Calculate pairwise squared distances
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            sq_dists = np.sum(diff**2, axis=2)
            np.fill_diagonal(sq_dists, np.inf) # Exclude self

            # Find indices of k nearest neighbors for each point
            actual_k = min(k_neighbors, n_points - 1) # Ensure k is not more than available neighbors
            neighbor_indices = np.argsort(sq_dists, axis=1)[:, :actual_k]

            # Add arrows as annotations
            for i in range(n_points):
                for k_idx in range(actual_k):
                    j = neighbor_indices[i, k_idx]
                    fig.add_annotation(
                        x=points[j, 0], y=points[j, 1], # Arrow end point (neighbor)
                        ax=points[i, 0], ay=points[i, 1], # Arrow start point (self)
                        xref='x', yref='y', axref='x', ayref='y',
                        showarrow=True,
                        arrowhead=2,       # Style of arrowhead
                        arrowsize=1,       # Size of arrowhead
                        arrowwidth=1,      # Width of arrow line
                        arrowcolor='rgba(0,0,255,0.5)' # Arrow color (semi-transparent blue)
                    )
        # --- End Arrow Section ---


        # Configure layout
        # Update plot title to reflect log density
        plot_title = f'Gaussian KDE Log Density (Power={power_value}, IncludeSelf={include_self_option}, k={k_neighbors})'
        fig.update_layout(
            title=plot_title,
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            yaxis_scaleanchor="x", # Ensure aspect ratio is 1:1
            xaxis_constrain="domain", # Ensure x-axis fills the domain
            yaxis_constrain="domain", # Ensure y-axis fills the domain
            margin=dict(l=40, r=40, b=40, t=80), # Adjust margins
            legend_title_text='Trace'
        )


        # 4. Save to HTML
        # Update filename to reflect log density and neighbors
        output_filename = f"kde_power_{power_value}_incSelf_{include_self_option}_k{k_neighbors}.html"
        fig.write_html(output_filename)
        if (diff := time.time() - start_time) > 1:
            print(f"Plotting took {diff:.2f} seconds.")
        print(f"Plot saved to {output_filename}")
        print("-" * 40 + "\n")

    print("All combinations processed.")