import pandas as pd
import numpy as np
import matplotlib.patches as patches
from scipy.stats import circmean, circstd, t
from matplotlib.patches import FancyArrowPatch, Ellipse, Wedge
from matplotlib.legend_handler import HandlerPatch
from matplotlib.transforms import Affine2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def extract_grid_distribution(pose_file, x_distances=0.01, y_distances=0.01):
    df_pose_raw = pd.read_csv(pose_file)

    # Drop row when Time value is 'Time'
    df_pose = df_pose_raw[df_pose_raw['Time'] != 'Time'].reset_index(drop=True)
    df_pose = df_pose.apply(pd.to_numeric)
    x_continuous = df_pose['PosX']
    y_continuous = df_pose['PosZ']
    azi_continuous = 360 - df_pose['RotY']    # From anti-clockwise to clockwise

    # Define grid boundaries
    x_bins = np.arange(-3.5, 3.5 + x_distances, x_distances)
    y_bins = np.arange(-3.5, 3.5 + y_distances, y_distances)
    grid_counts = np.zeros((len(x_bins)-1, len(y_bins)-1))

    # Count occurrences in each grid cell
    for x, y, in zip(x_continuous, y_continuous):
        x_idx = np.digitize(x, x_bins) - 1
        y_idx = np.digitize(y, y_bins) - 1
        if 0 <= x_idx < grid_counts.shape[0] and 0 <= y_idx < grid_counts.shape[1]:
            grid_counts[x_idx, y_idx] += 1

    # Normalise
    grid_counts /= np.sum(grid_counts)

    return grid_counts

def extract_path_efficiency(pose_file):
    df_pose_raw = pd.read_csv(pose_file)

    # Drop row when Time value is 'Time'
    df_pose = df_pose_raw[df_pose_raw['Time'] != 'Time'].reset_index(drop=True)
    df_pose = df_pose.apply(pd.to_numeric)
    x_continuous = df_pose['PosX']
    y_continuous = df_pose['PosZ']

    # Calculate difference between positions
    x_diff = np.diff(x_continuous)
    y_diff = np.diff(y_continuous)

    # Compute Euclidean distance
    distances = np.sqrt(x_diff**2 + y_diff**2)

    # Calculate total path length
    total_path_length = np.sum(distances)

    # Calculate straight-line distance from start to end
    straight_line_distance = np.sqrt((x_continuous.iloc[-1] - x_continuous.iloc[0])**2 + 
                                     (y_continuous.iloc[-1] - y_continuous.iloc[0])**2)
    
    # Calculate path efficiency
    path_efficiency = straight_line_distance / total_path_length if total_path_length > 0 else 0

    return path_efficiency

def extract_sweetspot(pose_file, last_n_samples=50):
    """
    Extract the sweet spot location and direction given a pose file

    Args:
        pose_file: The csv file name containing the pose data frame
        last_n_samples: The number of samples toward the end at which to compute the median for the sweet spot

    Returns:
        ss_xy: The sweet spot location (x, y)
        ss_azi: The sweet spot direction (in degrees)
    """

    df_pose_raw = pd.read_csv(pose_file)

    # Drop row when Time value is 'Time'
    df_pose = df_pose_raw[df_pose_raw['Time'] != 'Time'].reset_index(drop=True)
    df_pose = df_pose.apply(pd.to_numeric)
    x_continuous = df_pose['PosX']
    y_continuous = df_pose['PosZ']
    azi_continuous = 360 - df_pose['RotY']    # From anti-clockwise to clockwise

    # Sweet spot and angle
    # Get the median of the last 50 samples (1 second)
    ss_xy = (x_continuous.iloc[-last_n_samples:].median(), y_continuous.iloc[-last_n_samples:].median())
    ss_azi = azi_continuous.iloc[-last_n_samples:].median()

    return ss_xy, ss_azi

def calculate_mahalanobis_map(sweetspots, sweetangles, grid_size=11, azi_res=5):
    """
    Calculate the mahalanobis map given sweet spot data points

    Args:
        sweetspots: The sweet spot locations in (x, y)
        sweetangles: The sweet spot directions in degrees
        grid_size: The spacing of the grid in the x- and y-coordinates
        azi_res: The resolution for the direction (in degrees)

    Returns:
        X: The x-coordinate values in the mesh grid
        Y: The y-coordinate values in the mesh grid
        Phi: The phi-coordinate values in the mesh grid
        Z_normalized: The normalized Mahalanobis score at each point in the mesh grid
    """

    # Convert angles into 4D space
    phi_rad = np.deg2rad(sweetangles + 90)
    sin_phi = np.sin(phi_rad)
    cos_phi = np.cos(phi_rad)

    # Construct 4D data matrix
    data_4d = np.vstack([sweetspots[:, 0], sweetspots[:, 1], sin_phi, cos_phi]).T

    # Compute statistics
    mean_4d = np.mean(data_4d, axis=0)
    cov_4d = np.cov(data_4d, rowvar=False)
    inv_cov_4d = np.linalg.inv(cov_4d)

    # Set up grid
    x_space = np.linspace(-2.5, 2.5, grid_size)
    y_space = np.linspace(-2.5, 2.5, grid_size)
    phi_space = np.linspace(0, 360, int(360/azi_res), endpoint=False)
    X, Y, Phi = np.meshgrid(x_space, y_space, phi_space, indexing='ij')
    grid_coords = np.vstack([
        X.ravel(), 
        Y.ravel(),
        np.sin(np.radians((Phi + 90).ravel())),
        np.cos(np.radians((Phi + 90).ravel()))]).T

    # Compute 4D Mahalanobis distances
    diff = grid_coords - mean_4d
    squared_mahalanobis = np.sum(diff @ inv_cov_4d * diff, axis=1)

    # Transform to a probability score
    Z = np.exp(-0.5 * (squared_mahalanobis.reshape(X.shape)))

    # Scale to [0, 1]
    Z_normalized = (Z - Z.min()) / (Z.max() - Z.min())

    return X, Y, Phi, Z_normalized

def extract_objective_mahalanobis(df_naviqual, response_name, sweetspot_threshold=0.8):
    """
    Calculate the mahalanobis map of an objective quality with their objective sweet spots

    Args:
        df_naviqual: The data frame containing the objective quality metric values
        response_name: The objective quality metric, i.e. 'snr' or 'locq'
        sweetspot_threshold: The threshold value to determine the objective sweet spot

    Returns:
        X_md: The x-coordinate values in the mesh grid
        Y_md: The y-coordinate values in the mesh grid
        Phi_md: The phi-coordinate values in the mesh grid
        Z_md: The normalized Mahalanobis score at each point in the mesh grid
    """

    # Set up grid
    unique_x = np.sort(df_naviqual['listener_x'].unique())
    unique_y = np.sort(df_naviqual['listener_y'].unique())
    unique_phi = np.sort(df_naviqual['listener_azi'].unique())
    X, Y, Phi = np.meshgrid(unique_x, unique_y, unique_phi, indexing='ij')
    df_sorted = df_naviqual.sort_values(by=['listener_x', 'listener_y', 'listener_azi'])
    grid_shape = (len(unique_x), len(unique_y), len(unique_phi))
    Z = df_sorted[response_name].values.reshape(grid_shape)

    # Normalize response
    Z_normalized = (Z - Z.min()) / (Z.max() - Z.min())

    # Find subset of data points exceeding threshold
    mask = Z_normalized > sweetspot_threshold
    matching_x = np.unique(X[mask])
    matching_y = np.unique(Y[mask])
    matching_x_grid, matching_y_grid = np.meshgrid(matching_x, matching_y, indexing='ij')
    matching_xy = np.column_stack((matching_x_grid.ravel(), matching_y_grid.ravel()))

    sweetspots = []
    sweetangles = []
    # Find subset of phi values for each (x, y) pair that exceed the threshold after re-normalization.
    for (x, y) in matching_xy:
        Z_subset = Z_normalized[(X == x) & (Y == y)]
        Phi_subset = Phi[(X == x) & (Y == y)]
        Z_subset_normalized = (Z_subset - Z_subset.min()) / (Z_subset.max() - Z_subset.min())
        phi_indices = Z_subset_normalized > sweetspot_threshold
        matching_phi = Phi_subset[phi_indices]
        sweetspots.extend([(x, y)] * len(matching_phi))
        sweetangles.extend(matching_phi)
    sweetspots = np.array(sweetspots)
    sweetangles = np.array(sweetangles)

    # Calculate mahalanobis map based on this subset of sweet spot locations and directions
    X_md, Y_md, Phi_md, Z_md = calculate_mahalanobis_map(sweetspots, sweetangles, grid_size=11, azi_res=5)

    return X_md, Y_md, Phi_md, Z_md

def extract_average_ci_sweetspots(sweetspots):
    N_participants = len(sweetspots)

    # Compute for the average and 95% confidence intervals of the sweet spots and angles
    average_sweetspot = np.mean(sweetspots, axis=0)
    std_sweetspot = np.std(sweetspots, axis=0, ddof=1)

    # Confidence intervals
    ci_sweetspot = (t.interval(0.95, df=N_participants-1, loc=average_sweetspot[0], 
                              scale=std_sweetspot[0] / np.sqrt(N_participants)), 
                    t.interval(0.95, df=N_participants-1, loc=average_sweetspot[1], 
                               scale=std_sweetspot[1] / np.sqrt(N_participants)))
    
    return average_sweetspot, ci_sweetspot

def extract_average_ci_sweetangles(sweetangles):
    N_participants = len(sweetangles)

    average_sweetangle = np.rad2deg(circmean(np.deg2rad(sweetangles)))
    std_sweetangle = np.rad2deg(circstd(np.deg2rad(sweetangles)))

    ci_sweetangle = t.interval(0.95, df=N_participants-1, loc=average_sweetangle, 
                               scale=std_sweetangle / np.sqrt(N_participants))
    
    return average_sweetangle, ci_sweetangle

def annotate_environment(ax, sweetspots, sweetangles, speech_xys_scene, speech_loudness_level, modality_color, label='Subjective sweet spots'):
    """
    Annotate the elements in the virtual environment

    Args:
        ax: The axis on the plot
        sweetspots: The sweet spot locations in (x, y)
        sweetangles: The sweet spot directions in degrees
        speech_xys_scene: The speech distractor sources in (x, y)
        speech_loudness_level: The loudness levels of the speech sources
        modality_color: To assign a color on the sweet spot markers
        label: The legend label on the sweet spot markers

    Returns:
        ax: The output axis
    """
    # Configure plot
    ax.set_xlim([-3.0, 3.0])
    ax.set_ylim([-3.2, 3.2])
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')

    # Annotate the band locations
    source_locs = [(-1.5, 3.0), (-0.5, 3.0), (0.5, 3.0), (1.5, 3.0)]
    source_names = ['bass', 'drums', 'vocals', 'other']
    source_colors = ['tab:blue', 'tab:green', 'tab:cyan', 'tab:purple']

    # Annotate band locations
    for i in range(len(source_locs)):
        ax.scatter(source_locs[i][0], source_locs[i][1], color=source_colors[i], s=50, marker='o')
        ax.text(source_locs[i][0], source_locs[i][1] - 0.2, source_names[i], fontsize=10, ha='center', va='center', color='black')

    # Plot speech locations
    for i, speech_xy in enumerate(speech_xys_scene):
        ax.scatter(speech_xy[0], speech_xy[1], color='tab:red', s=50, marker='o')
        ax.text(speech_xy[0], speech_xy[1] + 0.2, f'{speech_loudness_level[i]} dB', fontsize=10, ha='center', va='center', color='white')
        ax.text(speech_xy[0], speech_xy[1] - 0.2, 'speech', fontsize=10, ha='center', va='center', color='white')

    # Annotate grid lines
    ax.vlines(x=[-1.5, -0.5, 0.5, 1.5], ymin=-2.5, ymax=2.5, color='black', linestyles='dashed', linewidth=0.5)
    ax.hlines(y=[-1.5, -0.5, 0.5, 1.5], xmin=-2.5, xmax=2.5, color='black', linestyles='dashed', linewidth=0.5)
    ax.vlines(x=[-2.5, 2.5], ymin=-2.5, ymax=2.5, color='black', linestyles='solid', linewidth=1.0)
    ax.hlines(y=[-2.5, 2.5], xmin=-2.5, xmax=2.5, color='black', linestyles='solid', linewidth=1.0)

    # Plot participant sweet spots and angles
    plt_sweetspots = ax.scatter(sweetspots[:, 0], sweetspots[:, 1], s=10, marker='*', 
                color=modality_color, alpha=0.7, label=label)
    plt_sweetdirections = ax.quiver(sweetspots[:, 0], sweetspots[:, 1], np.cos(np.deg2rad(90 + sweetangles)), np.sin(np.deg2rad(90 + sweetangles)), 
                angles='xy', scale_units='xy', alpha=0.7, scale=5, width=0.004, color=modality_color)
    return ax

def annotate_compass(ax, angle):
    """
    Annotate a compass indicating the listener direction on the contour map

    Args:
        ax: The contour map axis
        angle: The listener direction (in degrees)

    Returns: 
        ax: The resulting contour map with the compass annotation
    """
   # 1. Create an absolute inset axis container so the compass stays perfectly round
    # It places a 1.2 x 1.2 inch container in the lower right (loc=4) with small padding
    ax_compass = inset_axes(ax, width="22%", height="22%", loc=4, borderpad=0)
    ax_compass.set_xlim(-1.2, 1.2)
    ax_compass.set_ylim(-1.2, 1.2)
    ax_compass.set_aspect('equal', adjustable='box')
    ax_compass.axis('off')  # Hide the box boundaries of the inset axis
    
    # 2. Draw the Outer Dial & Border Ring
    dial = patches.Circle((0, 0), 1.0, edgecolor='#2c3e50', facecolor='#f8f9fa', linewidth=2.5, zorder=1)
    ax_compass.add_patch(dial)
    
    # 3. Add Cardinal Markers (N, S, E, W)
    ax_compass.text(0, 0.75, 'N', ha='center', va='center', weight='bold', color='#c0392b', fontsize=10)
    ax_compass.text(0, -0.75, 'S', ha='center', va='center', weight='bold', color='#2c3e50', fontsize=10)
    ax_compass.text(0.75, 0, 'E', ha='center', va='center', weight='bold', color='#2c3e50', fontsize=10)
    ax_compass.text(-0.75, 0, 'W', ha='center', va='center', weight='bold', color='#2c3e50', fontsize=10)
    
    # 4. Offset target angle by +90 degrees so 0 degrees points straight North
    # As 'frame' increases, the rotation follows a true Counter-Clockwise path
    compass_angle = angle
    
    # 5. Build the Dual-Colored Needle Geometry
    # Define local coordinates for North (pointing up) and South (pointing down) triangles
    north_pointer = np.array([[0, 0.7], [0.12, 0], [-0.12, 0]])
    south_pointer = np.array([[0, -0.7], [0.12, 0], [-0.12, 0]])
    
    # Apply standard 2D rotation matrix transformation using Matplotlib's transform tools
    rotation_transform = Affine2D().rotate_deg(compass_angle) + ax_compass.transData
    
    # Render the Red North Needle
    needle_n = patches.Polygon(north_pointer, facecolor='#e74c3c', edgecolor='#c0392b', linewidth=1, zorder=3)
    needle_n.set_transform(rotation_transform)
    ax_compass.add_patch(needle_n)
    
    # Render the Dark Grey South Needle
    needle_s = patches.Polygon(south_pointer, facecolor='#7f8c8d', edgecolor='#2c3e50', linewidth=1, zorder=3)
    needle_s.set_transform(rotation_transform)
    ax_compass.add_patch(needle_s)
    
    # Center Pivot Pin (The brass/silver cap holding the needle)
    pivot = patches.Circle((0, 0), 0.06, facecolor='white', edgecolor='#2c3e50', linewidth=1.5, zorder=4)
    ax_compass.add_patch(pivot)

    return ax

class CustomArrowHandler(HandlerPatch):
    def __init__(self, arrowstyle='-|>', mutation_scale=15, lw=2):
        super().__init__()
        self.arrowstyle = arrowstyle
        self.mutation_scale = mutation_scale
        self.lw = lw

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        p = FancyArrowPatch((0, height/2), (width, height/2), 
                            arrowstyle=self.arrowstyle, 
                            mutation_scale=self.mutation_scale, 
                            linewidth=self.lw, color=orig_handle.get_facecolors()[0])
        p.set_transform(trans)
        return [p]

class CustomEllipseHandler(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Center the ellipse in the legend slot space
        p = Ellipse((width/2, height/2), width=width*0.8, height=height*0.6, 
                    facecolor='none', edgecolor=orig_handle.get_edgecolor(), lw=2)
        p.set_transform(trans)
        return [p]

class CustomWedgeHandler(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Draw a wedge (like a pie slice) centered in the legend slot
        # Parameters: (center_x, center_y), radius, theta1 (start angle), theta2 (end angle)
        p = Wedge((width/2, height/2 - 2), r=height, theta1=30, theta2=150, 
                  facecolor=orig_handle.get_facecolor(), edgecolor=orig_handle.get_edgecolor())
        p.set_transform(trans)
        return [p]