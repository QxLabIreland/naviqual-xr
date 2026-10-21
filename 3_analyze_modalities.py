import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import math
import pingouin as pg
from scipy.stats import circmean, circstd, t
from matplotlib.patches import Ellipse, Wedge
from matplotlib.legend_handler import HandlerPatch
from src.utils import *

# Save to output folder
plots_folder = f'./output/3_cue_modalities/'
os.makedirs(plots_folder, exist_ok=True)

# Scene setup
speech_xys_scenes = [[(0, -2)], [(2, 0)], 
                    [(0, -2), (2, 2)], [(0, -2), (-2, 2)],
                    [(2, 2), (0, 0), (-2, -2)], [(2, 2), (0, 0), (-2, -2)],
                    [(0, -2)], [(2, 0)], 
                    [(0, -2), (2, 2)], [(0, -2), (-2, 2)],
                    [(2, 2), (0, 0), (-2, -2)], [(2, 2), (0, 0), (-2, -2)]]
speech_loudness_levels = [[0, 0, 0], [0, 0, 0], 
                        [0, 0, 0], [0, 0, 0],
                        [0, 0, 0], [0, -5, -10],
                        [0, 0, 0], [0, 0, 0], 
                        [0, 0, 0], [0, 0, 0],
                        [0, 0, 0], [0, -5, -10]]

# Data to store for each scene
distances_scenes = []
angular_diffs_scenes = []
iou_ellipses = []
iou_cones = []

# Iterate over auditory scenes
for scene_index in range(6):
    print(f'Analysing scene S{scene_index+1}...')
    speech_xys_scene = speech_xys_scenes[scene_index]
    speech_loudness_level = speech_loudness_levels[scene_index]

    # Data to store for each participant
    auditory_sweetspots = []
    auditory_sweetangles = []
    audiovisual_sweetspots = []
    audiovisual_sweetangles = []

    for participant_id in range(25):
        pose_directory = f'./data/pose/participant{participant_id+1:02d}/'
        for file in os.listdir(pose_directory):
            if f'Arena {scene_index+1}.csv' in file:
                auditory_pose_file = f'{pose_directory}{file}'
            elif f'Arena {scene_index+7}.csv' in file:
                audiovisual_pose_file = f'{pose_directory}{file}'

        # Extract sweet spot and sweet angle
        auditory_sweetspot, auditory_sweetangle = extract_sweetspot(auditory_pose_file)
        audiovisual_sweetspot, audiovisual_sweetangle = extract_sweetspot(audiovisual_pose_file)

        # Store data for each participant
        auditory_sweetspots.append(auditory_sweetspot)
        auditory_sweetangles.append(auditory_sweetangle)
        audiovisual_sweetspots.append(audiovisual_sweetspot)
        audiovisual_sweetangles.append(audiovisual_sweetangle)
    
    # Convert lists to numpy arrays
    auditory_sweetspots = np.array(auditory_sweetspots)
    auditory_sweetangles = np.array(auditory_sweetangles)
    audiovisual_sweetspots = np.array(audiovisual_sweetspots)
    audiovisual_sweetangles = np.array(audiovisual_sweetangles)
    N_participants = len(auditory_sweetspots)

    # Initialise the plot
    fig, ax = plt.subplots()
    ax.axis('scaled')
    ax.set_title(f'S{scene_index+1} Sweet spot locations and directions across modalities')
    ax.set_xlim([-3.0, 3.0])
    ax.set_ylim([-3.0, 3.2])
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
        ax.text(speech_xy[0], speech_xy[1] + 0.2, f'{speech_loudness_level[i]} dB', fontsize=10, ha='center', va='center', color='black')
        ax.text(speech_xy[0], speech_xy[1] - 0.2, 'speech', fontsize=10, ha='center', va='center')

    # Annotate grid lines
    ax.vlines(x=[-1.5, -0.5, 0.5, 1.5], ymin=-2.5, ymax=2.5, color='black', linestyles='dashed', linewidth=0.5)
    ax.hlines(y=[-1.5, -0.5, 0.5, 1.5], xmin=-2.5, xmax=2.5, color='black', linestyles='dashed', linewidth=0.5)
    ax.vlines(x=[-2.5, 2.5], ymin=-2.5, ymax=2.5, color='black', linestyles='solid', linewidth=1.0)
    ax.hlines(y=[-2.5, 2.5], xmin=-2.5, xmax=2.5, color='black', linestyles='solid', linewidth=1.0)

    # Plot auditory sweet spots and angles
    ax.scatter(auditory_sweetspots[:, 0], auditory_sweetspots[:, 1], s=10, marker='*', 
                color='blue', alpha=0.7, label='Auditory sweet spots')
    for ss_xy, ss_azi in zip(auditory_sweetspots, auditory_sweetangles):
        ax.quiver(ss_xy[0], ss_xy[1], np.cos(np.deg2rad(90 + ss_azi)), np.sin(np.deg2rad(90 + ss_azi)), 
                    angles='xy', scale_units='xy', alpha=0.7, scale=5, width=0.004, color='blue')

    # Plot audiovisual sweet spots and angles
    ax.scatter(audiovisual_sweetspots[:, 0], audiovisual_sweetspots[:, 1], s=10, marker='*', 
                color='magenta', alpha=0.7, label='Audio-visual sweet spots')
    for ss_xy, ss_azi in zip(audiovisual_sweetspots, audiovisual_sweetangles):
        ax.quiver(ss_xy[0], ss_xy[1], np.cos(np.deg2rad(90 + ss_azi)), np.sin(np.deg2rad(90 + ss_azi)), 
                    angles='xy', scale_units='xy', alpha=0.7, scale=5, width=0.004, color='magenta')

    # Connect the sweet spots
    for aud_ss, av_ss in zip(auditory_sweetspots, audiovisual_sweetspots):
        ax.plot([aud_ss[0], av_ss[0]], [aud_ss[1], av_ss[1]], color='black', alpha=0.5, linewidth=0.5)

    # Compute for the average and 95% confidence intervals of the sweet spots and angles
    a_average_sweetspot, a_ci_sweetspot = extract_average_ci_sweetspots(auditory_sweetspots)
    a_average_sweetangle, a_ci_sweetangle = extract_average_ci_sweetangles(auditory_sweetangles)
    av_average_sweetspot, av_ci_sweetspot = extract_average_ci_sweetspots(audiovisual_sweetspots)
    av_average_sweetangle, av_ci_sweetangle = extract_average_ci_sweetangles(audiovisual_sweetangles)
    
    # Plot average sweet spot and angle for auditory scenes
    ax.scatter(a_average_sweetspot[0], a_average_sweetspot[1], s=100, color='blue', 
                marker='*')
    ax.quiver(a_average_sweetspot[0], a_average_sweetspot[1], np.cos(np.deg2rad(90 + a_average_sweetangle)), 
               np.sin(np.deg2rad(90 + a_average_sweetangle)), color='blue', angles='xy', 
               scale_units='xy', scale=3, width=0.005)

    # Draw an ellipse to represent the sweet spot confidence intervals
    a_ellipse = Ellipse(xy=a_average_sweetspot, width=a_ci_sweetspot[0][1]-a_ci_sweetspot[0][0], 
                      height=a_ci_sweetspot[1][1]-a_ci_sweetspot[1][0], edgecolor='blue', 
                      fc='None', lw=2)
    ax.add_patch(a_ellipse)

    # Draw a cone to represent the sweet angle confidence intervals\
    a_cone = Wedge(center=a_average_sweetspot, r=0.5, theta1=90 + a_ci_sweetangle[0], 
                 theta2=90 + a_ci_sweetangle[1], color='blue')
    ax.add_patch(a_cone)

    # Plot average sweet spot and angle for audiovisual scenes
    ax.scatter(av_average_sweetspot[0], av_average_sweetspot[1], s=100, color='magenta', 
                marker='*')
    ax.quiver(av_average_sweetspot[0], av_average_sweetspot[1], np.cos(np.deg2rad(90 + av_average_sweetangle)), 
               np.sin(np.deg2rad(90 + av_average_sweetangle)), color='magenta', angles='xy', 
               scale_units='xy', scale=3, width=0.005)

    # Draw an ellipse to represent the sweet spot confidence intervals
    av_ellipse = Ellipse(xy=av_average_sweetspot, width=av_ci_sweetspot[0][1]-av_ci_sweetspot[0][0], 
                      height=av_ci_sweetspot[1][1]-av_ci_sweetspot[1][0], edgecolor='magenta', 
                      fc='None', lw=2)
    ax.add_patch(av_ellipse)

    # Draw a cone to represent the sweet angle confidence intervals\
    av_cone = Wedge(center=av_average_sweetspot, r=0.5, theta1=90 + av_ci_sweetangle[0], 
                 theta2=90 + av_ci_sweetangle[1], color='magenta')
    ax.add_patch(av_cone)

    # Save figures
    ax.legend(loc='lower center', ncol=2,fontsize=8)

    plt.savefig(f'{plots_folder}S{scene_index+1}_modalities.pdf', dpi=300, bbox_inches='tight')
    plt.close()