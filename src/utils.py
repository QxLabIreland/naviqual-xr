import pandas as pd
import numpy as np

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