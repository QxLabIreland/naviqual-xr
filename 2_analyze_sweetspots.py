import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pingouin as pg
from scipy.stats import circmean, circstd, t
from matplotlib.patches import Ellipse, Wedge
from src.utils import *

# Save to output folder
plots_folder = f'./output/2_participant_sweetspots/'
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
sweetspots_per_scene = []
sweetangles_per_scene = []

# Iterate over all scenes
for scene_index in range(12):
    cue_modality = 'A' if scene_index < 6 else 'AV'

    # Set color depending on modality: magenta = auditory, blue = audio-visual
    modality_color = 'blue' if cue_modality == 'A' else 'magenta'

    print(f'Analysing scene S{(scene_index%6)+1}-{cue_modality}...')
    speech_xys_scene = speech_xys_scenes[scene_index]
    speech_loudness_level = speech_loudness_levels[scene_index]

    # Data to store for each participant
    sweetspots = []
    sweetangles = []

    for participant_id in range(25):
        pose_directory = f'./data/pose/participant{participant_id+1:02d}/'
        for file in os.listdir(pose_directory):
            if f'Arena {scene_index+1}.csv' in file:
                pose_file = f'{pose_directory}{file}'

        # Extract sweet spot and sweet angle
        sweetspot, sweetangle = extract_sweetspot(pose_file)

        # Store data for each participant
        sweetspots.append(sweetspot)
        sweetangles.append(sweetangle)
    
    # Convert lists to numpy arrays
    sweetspots = np.array(sweetspots)
    sweetangles = np.array(sweetangles)

    N_participants = len(sweetspots)

    # Initialise the plot
    fig, ax = plt.subplots()
    ax.axis('scaled')
    ax.set_title(f'S{(scene_index%6)+1}-{cue_modality} Participant sweet spot locations and directions')
    ax.set_xlim([-3.0, 3.0])
    ax.set_ylim([-3.3, 3.2])
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

    # Plot participant sweet spots and angles
    plt_sweetspots = ax.scatter(sweetspots[:, 0], sweetspots[:, 1], s=10, marker='*', 
                color=modality_color, alpha=0.7, label='Sweet spot locations')
    plt_sweetdirections = ax.quiver(sweetspots[:, 0], sweetspots[:, 1], np.cos(np.deg2rad(90 + sweetangles)), np.sin(np.deg2rad(90 + sweetangles)), 
                angles='xy', scale_units='xy', alpha=0.7, scale=5, width=0.004, color=modality_color, 
                label='Sweet spot directions')
    
    # Compute for the average and 95% confidence intervals of the sweet spots and angles
    average_sweetspot = np.mean(sweetspots, axis=0)
    std_sweetspot = np.std(sweetspots, axis=0, ddof=1)
    average_sweetangle = np.rad2deg(circmean(np.deg2rad(sweetangles)))
    std_sweetangle = np.rad2deg(circstd(np.deg2rad(sweetangles)))

    # Confidence intervals
    ci_sweetspot = (t.interval(0.95, df=N_participants-1, loc=average_sweetspot[0], 
                              scale=std_sweetspot[0] / np.sqrt(N_participants)), 
                    t.interval(0.95, df=N_participants-1, loc=average_sweetspot[1], 
                               scale=std_sweetspot[1] / np.sqrt(N_participants)))
    ci_sweetangle = t.interval(0.95, df=N_participants-1, loc=average_sweetangle, 
                               scale=std_sweetangle / np.sqrt(N_participants))
    
    # Plot average sweet spot and angle
    ax.scatter(average_sweetspot[0], average_sweetspot[1], s=100, color=modality_color, 
                marker='*')
    ax.quiver(average_sweetspot[0], average_sweetspot[1], np.cos(np.deg2rad(90 + average_sweetangle)), 
               np.sin(np.deg2rad(90 + average_sweetangle)), color=modality_color, angles='xy', 
               scale_units='xy', scale=3, width=0.005)

    # Draw an ellipse to represent the sweet spot confidence intervals
    ellipse = Ellipse(xy=average_sweetspot, width=ci_sweetspot[0][1]-ci_sweetspot[0][0], 
                      height=ci_sweetspot[1][1]-ci_sweetspot[1][0], edgecolor=modality_color, 
                      fc='None', lw=2, label='Consensus location')
    ax.add_patch(ellipse)

    # Draw a cone to represent the sweet angle confidence intervals\
    cone = Wedge(center=average_sweetspot, r=0.5, theta1=90 + ci_sweetangle[0], 
                 theta2=90 + ci_sweetangle[1], color=modality_color, 
                 label='Consensus direction')
    ax.add_patch(cone)

    # Save figures
    ax.legend(handles=[plt_sweetspots, plt_sweetdirections, ellipse, cone], 
               labels=['Sweet spot locations', 'Sweet spot directions', 'Consensus location', 'Consensus direction'], 
               handler_map={type(plt_sweetdirections): CustomArrowHandler(arrowstyle='-|>', mutation_scale=12), 
                            type(ellipse): CustomEllipseHandler(), 
                            type(cone): CustomWedgeHandler()}, 
               loc='lower center', ncol=2, fontsize=8)
    plt.savefig(f'{plots_folder}S{(scene_index%6)+1}-{cue_modality}_sweetspots.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # Store data for each scene
    sweetspots_per_scene.append(sweetspots)
    sweetangles_per_scene.append(sweetangles)

sweetspots_per_scene = np.array(sweetspots_per_scene)
sweetangles_per_scene = np.array(sweetangles_per_scene)

# Statistical analysis
# Create a long data frame
df_list = []
for scene_index in range(12):
    modality = 'A' if scene_index < 6 else 'AV'
    environment_name = 'Auditory' if scene_index < 6 else 'Audio-Visual'
    scene_number = (scene_index % 6) + 1
    
    sweetspots = sweetspots_per_scene[scene_index, :, :]
    sweetangles = sweetangles_per_scene[scene_index, :]
    
    for i, (sweetspot, sweetangle) in enumerate(zip(sweetspots, sweetangles)):
        df_list.append({'Participant': f'P{i+1:02d}',
                        'Sweet Spot X Location': sweetspot[0],
                        'Sweet Spot Y Location': sweetspot[1],
                        'Sweet Spot Direction': sweetangle,
                        'Scene': f'S{scene_number}',
                        'Modality': environment_name})
        
df = pd.DataFrame(df_list)

# Save data frame
df.to_csv(f'{plots_folder}participant_sweetspots.csv', index=False)

# Load data frame
df = pd.read_csv(f'{plots_folder}participant_sweetspots.csv')

# Convert angles from 0 to 360 to -180 to 180
df['Sweet Spot Direction'] = ((df['Sweet Spot Direction'] + 180) % 360) - 180

# Run two-way repeated measures anova 
# Confirm whether X-coordinates are significantly affected
anova_results = pg.rm_anova(
    data=df,
    dv='Sweet Spot X Location',
    within=['Scene', 'Modality'],
    subject='Participant', 
    detailed=True,
)
print(anova_results)

# Run post-hoc tests
post_hocs = pg.pairwise_tests(
    data=df, 
    dv='Sweet Spot X Location', 
    within=['Scene', 'Modality'],
    subject='Participant',
    alpha=0.05, 
    alternative='less',
)
print(post_hocs)

# Confirm whether Y-coordinates are significantly affected
anova_results = pg.rm_anova(
    data=df,
    dv='Sweet Spot Y Location',
    within=['Scene', 'Modality'],
    subject='Participant', 
    detailed=True,
)
print(anova_results)

# Confirm whether sweet spot directions are significantly affected
anova_results = pg.rm_anova(
    data=df,
    dv='Sweet Spot Direction',
    within=['Scene', 'Modality'],
    subject='Participant', 
    detailed=True,
)
print(anova_results)
