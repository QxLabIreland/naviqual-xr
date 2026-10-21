import pandas as pd
import numpy as np
from scipy.stats import circmean, circstd, t
from matplotlib.patches import FancyArrowPatch, Ellipse, Wedge
from matplotlib.legend_handler import HandlerPatch

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

def extract_sweetspot_angle(pose_file, last_n_samples=50):
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