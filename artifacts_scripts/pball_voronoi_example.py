import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection # Added this import
import sys
from pathlib import Path

# --- Parameters ---
NUM_ARTICLES = 200
NUM_BOOKS = 20
BOX_SIZE = 6.5  # Defines the coordinate space (e.g., 0.0 to 10.0)
RANDOM_SEED = 1 # Added for reproducibility
QUERY_LOCATION = np.array([0.6, 0.2]) * BOX_SIZE

# --- GMM Sampling Configuration ---
# We define centers (means) and covariance matrices for our mixtures.
# Each center is [x, y], each cov is [[var_x, cov_xy], [cov_xy, var_y]]
# We'll create one "pure" article blob (top-right), one "pure" book blob (bottom-left),
# and one "mixed" area (top-left) where they overlap.

BOOK_CENTERS = [ [2, 2], [2.5, 4.5] ] # Bottom-left, Top-left (overlap)
BOOK_COVS = [ [[0.6, 0.1], [0.1, 0.8]],   # Tilts slightly up and to the right
                [[0.7, -0.3], [-0.3, 0.7]] ] # Tilts down and to the right

ARTICLE_CENTERS = [ [5, 2] ] # Top-right, Top-left
ARTICLE_COVS = [ [[0.7, 0.3], [0.3, 0.5]] ] # Tilts down and to the right

# Make the book gaussians a subset of the article gaussians
ARTICLE_CENTERS += BOOK_CENTERS
ARTICLE_COVS += BOOK_COVS

# ---------------------------------

# --- Style Configuration ---
# Matplotlib Colors
ARTICLE_COLOR_MPL = 'black'
BOOK_COLOR_MPL = 'blue'
QUERY_COLOR_MPL = 'green'

# LaTeX/TikZ Colors
ARTICLE_COLOR_TEX = 'black'
BOOK_COLOR_TEX = 'blue'
QUERY_COLOR_TEX = 'green!60!black' # A darker green for LaTeX

ARTICLE_SIZE_MPL = 5 # Matplotlib size (area) for articles
BOOK_SIZE_MPL = 25   # Matplotlib size (area) for books (will be hollow)
QUERY_SIZE_MPL = 30  # Matplotlib size (area) for query

BOOK_LINEWIDTH_MPL = 1.2 # Linewidth for hollow book markers

ARTICLE_SIZE_TEX = 1.0 # LaTeX radius for articles (in pt)
BOOK_SIZE_TEX = 3.0    # LaTeX radius for books (in pt)
QUERY_SIZE_TEX = 3.5   # LaTeX radius for query (in pt)

BOOK_LINEWIDTH_TEX = 'thick' # Linewidth for hollow book markers ('thin', 'thick', 'very thick')
# --------------------

def sample_gmm_clamped(num_points, centers, covs, box_size):
    """
    Samples points from a Gaussian Mixture Model (GMM) using rejection sampling
    to keep all points within the [0, box_size] bounds.
    
    Args:
        num_points (int): The final number of points required.
        centers (list): A list of [x, y] mean coordinates for each Gaussian.
        covs (list): A list of 2x2 covariance matrices for each Gaussian.
        box_size (float): The upper bound for x and y (lower is 0).

    Returns:
        np.array: An array of (num_points, 2) coordinates.
    """
    points = np.empty((0, 2))
    num_centers = len(centers)
    
    # We'll over-sample slightly in each batch, as some will be rejected.
    # This is much faster than sampling one point at a time.
    batch_size = int(num_points * 1.1) + 5 

    while points.shape[0] < num_points:
        # 1. Decide which center to sample from for each point in the batch
        chosen_indices = np.random.choice(num_centers, batch_size)
        
        # 2. Generate samples from the chosen Gaussians
        new_samples = np.empty((0, 2))
        for i in range(num_centers):
            num_from_this_blob = np.sum(chosen_indices == i)
            if num_from_this_blob > 0:
                mean = centers[i]
                cov = covs[i]
                samples = np.random.multivariate_normal(mean, cov, num_from_this_blob)
                new_samples = np.vstack([new_samples, samples])

        # 3. Rejection Sampling: Keep only points inside the box
        in_bounds_x = (new_samples[:, 0] >= 0) & (new_samples[:, 0] <= box_size)
        in_bounds_y = (new_samples[:, 1] >= 0) & (new_samples[:, 1] <= box_size)
        
        in_bounds = new_samples[in_bounds_x & in_bounds_y]
        
        # 4. Add the valid points to our collection
        points = np.vstack([points, in_bounds])

    # 5. We likely have more than num_points, so trim and return
    return points[:num_points, :]

def get_voronoi_lines(vor):
    """
    Uses voronoi_plot_2d to handle infinite regions and returns
    a list of line segments [((x1, y1), (x2, y2)), ...].
    
    This is a robust way to get the line segments for plotting,
    as scipy's plotting function correctly handles the ridges
    that extend to infinity.
    """
    # Create a figure in the background, but don't show it
    # We create fig and ax separately so we can set the limits
    # *before* extracting the line data.
    fig, ax = plt.subplots()
    
    # Plot the voronoi diagram onto this ax
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='gray',
                          line_width=1, line_alpha=0.6, point_size=0)
    
    # CRITICAL: Set the limits to match our desired box.
    # This forces matplotlib to clip the lines before we extract them.
    ax.set_xlim(0, BOX_SIZE)
    ax.set_ylim(0, BOX_SIZE)
    
    line_segments = []
    
    # --- BUG FIX ---
    # The plot lines are a LineCollection in ax.collections,
    # not individual Line2D objects in ax.lines (which is ax.get_lines()).
    
    # We must iterate through all collections on the axis
    for collection in ax.collections:
        # Check if this collection is a LineCollection
        if isinstance(collection, LineCollection):
            # If so, get all line segments from it
            for segment in collection.get_segments():
                # segment is an array of two points [ [x1, y1], [x2, y2] ]
                # Convert to the tuple format we use
                line_segments.append(( (segment[0][0], segment[0][1]), 
                                       (segment[1][0], segment[1][1]) ))
    
    plt.close(fig) # Close the (unshown) figure
    return line_segments

def generate_plot_data():
    """Generates all the coordinates and Voronoi data."""
    # Set the random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    # 1. Generate coordinates using GMM sampler
    article_dots = sample_gmm_clamped(NUM_ARTICLES, ARTICLE_CENTERS, ARTICLE_COVS, BOX_SIZE)
    book_dots = sample_gmm_clamped(NUM_BOOKS, BOOK_CENTERS, BOOK_COVS, BOX_SIZE)
    
    # Use the fixed query dot location from the config
    query_dot = np.array(QUERY_LOCATION)

    # 2. Find nearest blue dot
    blue_tree = KDTree(book_dots)
    radius, _ = blue_tree.query(query_dot)

    # 3. Compute Voronoi diagram
    voronoi_points = np.vstack([book_dots, query_dot])
    vor = Voronoi(voronoi_points)
    
    return article_dots, book_dots, query_dot, radius, vor

def generate_latex_figure(articles, books, query, radius, vor, standalone=True):
    """
    Generates the complete LaTeX (TikZ) string for the figure.
    
    Args:
        ...
        standalone (bool): If True, wraps the TikZ code in a full
                           standalone LaTeX document. If False,
                           returns only the tikzpicture environment.
    """
    
    # 1. Get all line segments for the Voronoi plot
    voronoi_lines_data = get_voronoi_lines(vor)

    # 2. Build the LaTeX string line by line
    latex_lines = []
    
    # --- LaTeX Preamble (Optional) ---
    if standalone:
        latex_lines.append(r"\documentclass[tikz,border=10pt]{standalone}")
        latex_lines.append(r"\usepackage{tikz}")
        # \usetikzlibrary{transparency} removed as it's not available
        latex_lines.append(r"\begin{document}")
    
    # --- TikZ Figure ---
    # Add a comment for the input-able file, if needed.
    if not standalone:
        latex_lines.append(r"% --- This file is designed to be \input{}'d into a main .tex document ---")
        latex_lines.append(r"% Ensure your main document preamble has: \usepackage{tikz}")
        
    latex_lines.append(r"\begin{tikzpicture}")
    
    # --- Bounding Box and Clipping ---
    # Set the drawing area and clip everything outside it
    latex_lines.append(r"% --- Set drawing boundary and clip ---")
    latex_lines.append(rf"\useasboundingbox (0,0) rectangle ({BOX_SIZE},{BOX_SIZE});")
    latex_lines.append(rf"\clip (0,0) rectangle ({BOX_SIZE},{BOX_SIZE});")

    # --- Draw Border ---
    latex_lines.append(r"% --- Draw border ---")
    latex_lines.append(rf"\draw[black, thin] (0,0) rectangle ({BOX_SIZE},{BOX_SIZE});")

    # --- Draw Voronoi Diagram ---
    latex_lines.append(r"% --- Voronoi Diagram (for Blue + Green dots) ---")
    for start, end in voronoi_lines_data:
        # Format each line segment as a TikZ draw command
        latex_lines.append(rf"\draw[gray, thin, opacity=0.8] ({start[0]},{start[1]}) -- ({end[0]},{end[1]});")

    # --- Draw Orange Dots ---
    latex_lines.append(r"% --- Orange Dots (scatter) ---")
    for x, y in articles:
        latex_lines.append(rf"\fill[{ARTICLE_COLOR_TEX}] ({x},{y}) circle ({ARTICLE_SIZE_TEX}pt);")

    # --- Draw Blue Dots ---
    latex_lines.append(r"% --- Blue Dots (Voronoi points) ---")
    for x, y in books:
        # Changed from \fill to \draw to make it a hollow circle
        latex_lines.append(rf"\draw[{BOOK_COLOR_TEX}, {BOOK_LINEWIDTH_TEX}] ({x},{y}) circle ({BOOK_SIZE_TEX}pt);")

    # --- Draw Green Dot and Circle ---
    latex_lines.append(r"% --- Green Dot (Voronoi point) and Nearest-Neighbor Circle ---")
    g_x, g_y = query
    # Make the green dot slightly larger to stand out
    latex_lines.append(rf"\fill[{QUERY_COLOR_TEX}] ({g_x},{g_y}) circle ({QUERY_SIZE_TEX}pt);")
    # Draw the dashed circle with the calculated radius
    latex_lines.append(rf"\draw[{QUERY_COLOR_TEX}, thick, dashed] ({g_x},{g_y}) circle ({radius});")

    # --- Draw Legend ---
    latex_lines.append(r"% --- Legend ---")
    # We place the legend in the top-right corner.
    # We use a 'scope' to create a local coordinate system.
    # The legend is placed at 98% x, 98% y of the box size.
    latex_lines.append(rf"\begin{{scope}}[shift={{({BOX_SIZE*0.98},{BOX_SIZE*0.98})}}, font=\scriptsize]")
    # A solid white box for the legend background
    # Removed fill-opacity=0.7 to avoid dependency on 'transparency' library
    latex_lines.append(r"\node[anchor=north east, fill=white, draw=gray, thin, inner sep=3pt] (legend) at (0,0) {")
    # Use a tabular environment for easy alignment
    latex_lines.append(r"  \begin{tabular}{ll}")
    # Query
    latex_lines.append(rf"  \tikz\fill[{QUERY_COLOR_TEX}] (0,0) circle ({QUERY_SIZE_TEX}pt); & Query \\")
    # Book
    latex_lines.append(rf"  \tikz\draw[{BOOK_COLOR_TEX}, {BOOK_LINEWIDTH_TEX}] (0,0) circle ({BOOK_SIZE_TEX}pt); & Book \\")
    # Article
    latex_lines.append(rf"  \tikz\fill[{ARTICLE_COLOR_TEX}] (0,0) circle ({ARTICLE_SIZE_TEX}pt); & Article \\")
    # Voronoi Line
    latex_lines.append(r"  \tikz\draw[gray, thin, opacity=0.8] (0,0) -- (0.4,0); & Voronoi Cell \\")
    # 1-ball
    latex_lines.append(rf"  \tikz\draw[{QUERY_COLOR_TEX}, thick, dashed] (0,0) circle ({QUERY_SIZE_TEX}pt); & Query 1-ball \\")
    
    latex_lines.append(r"  \end{tabular}")
    latex_lines.append(r"};")
    latex_lines.append(r"\end{scope}")

    # --- End of file ---
    latex_lines.append(r"\end{tikzpicture}")
    
    if standalone:
        latex_lines.append(r"\end{document}")
    
    return "\n".join(latex_lines)

def save_matplotlib_figure(article_dots, book_dots, query_dot, radius, vor, output_path):
    """
    Generates and saves a matplotlib figure of the plot.
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # 1. Draw Voronoi diagram
    # Use the provided function, but customize it
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='gray',
                    line_width=1, line_alpha=0.8, point_size=0)

    # 2. Draw Orange Dots
    ax.scatter(article_dots[:, 0], article_dots[:, 1], c=ARTICLE_COLOR_MPL, s=ARTICLE_SIZE_MPL, label='Articles')

    # 3. Draw Blue Dots
    # Changed to hollow circles (facecolors='none') with an edge
    ax.scatter(book_dots[:, 0], book_dots[:, 1], 
               facecolors='none', 
               edgecolors=BOOK_COLOR_MPL, 
               s=BOOK_SIZE_MPL, 
               linewidths=BOOK_LINEWIDTH_MPL, 
               label='Books')
    
    # 4. Draw Green Dot and Circle
    g_x, g_y = query_dot
    ax.scatter([g_x], [g_y], c=QUERY_COLOR_MPL, s=QUERY_SIZE_MPL, label='Query', zorder=5)
    
    # Add the circle patch
    # Updated label to 'Query 1-ball'
    circle = plt.Circle((g_x, g_y), radius, color=QUERY_COLOR_MPL, fill=False, 
                        linestyle='--', linewidth=1.5, label='Query 1-ball')
    ax.add_patch(circle)

    # 5. Set bounds and style
    ax.set_xlim(0, BOX_SIZE)
    ax.set_ylim(0, BOX_SIZE)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize='small')
    ax.set_title('Voronoi Diagram of Books with Query')

    # 6. Omit axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # 7. Save the figure
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig) # Close the figure to free memory


if __name__ == "__main__":
    # Ensure all required libraries are installed
    try:
        import numpy
        import scipy
        import matplotlib
    except ImportError as e:
        print(f"Error: Missing required library. Please install {e.name}.")
        print("You can install them with: pip install numpy scipy matplotlib")
        sys.exit(1)

    # --- Setup Output Directory ---
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    
    # --- Generate Data ---
    print("Generating plot data...")
    article_dots, book_dots, query_dot, radius, vor = generate_plot_data()

    # # --- 1. Generate and Save LaTeX Figure (Standalone) ---
    # print("Generating standalone LaTeX figure...")
    # latex_output_standalone = generate_latex_figure(article_dots, book_dots, query_dot, radius, vor, standalone=True)
    
    # tex_output_path_standalone = output_dir / "voronoi_figure.tex"
    # try:
    #     # Write the content to a .tex file
    #     with open(tex_output_path_standalone, "w") as f:
    #         f.write(latex_output_standalone)
        
    #     print(f"Successfully generated '{tex_output_path_standalone}'")
    
    # except IOError as e:
    #     print(f"Error: Could not write to file '{tex_output_path_standalone}'.")
    #     print(f"Details: {e}")

    # --- 2. Generate and Save LaTeX Figure (Input-able) ---
    print("Generating input-able LaTeX figure...")
    latex_output_input = generate_latex_figure(article_dots, book_dots, query_dot, radius, vor, standalone=False)
    
    tex_output_path_input = output_dir / "voronoi_figure_input.tex"
    try:
        # Write the content to a .tex file
        with open(tex_output_path_input, "w") as f:
            f.write(latex_output_input)
        
        print(f"Successfully generated '{tex_output_path_input}'")
    
    except IOError as e:
        print(f"Error: Could not write to file '{tex_output_path_input}'.")
        print(f"Details: {e}")


    # --- 3. Generate and Save Matplotlib Figure (PNG) ---
    print("Generating Matplotlib (PNG) figure...")
    png_output_path = output_dir / "voronoi_figure.png"
    try:
        save_matplotlib_figure(article_dots, book_dots, query_dot, radius, vor, png_output_path)
        print(f"Successfully generated '{png_output_path}'")
    except Exception as e:
        print(f"Error: Could not generate matplotlib figure.")
        print(f"Details: {e}")

    print("\nDone.")