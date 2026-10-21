import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pingouin as pg
from src.utils import *

# Save to output folder
plots_folder = f'./output/1_navigation_behaviour/'
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
start_points = [(-1, -2), (2, -1), (1, -2), (-1, -2), (-1, -2), (-1, -2), 
                (-1, -2), (2, -1), (1, -2), (-1, -2), (-1, -2), (-1, -2)]
start_markers = ['>', '^', '<', '>', '<', '<', 
                 '>', '^', '<', '>', '<', '<']

# Data to store for each scene
path_efficiencies_per_scene = []

# Iterate over all scenes
for scene_index in range(12):
    cue_modality = 'A' if scene_index < 6 else 'AV'
    environment = 'Auditory' if cue_modality == 'A' else 'Audio-visual'

    # Set color depending on modality: magenta = auditory, blue = audio-visual
    modality_color = 'blue' if cue_modality == 'A' else 'magenta'

    print(f'Analysing scene S{(scene_index%6)+1}-{cue_modality}...')
    speech_xys_scene = speech_xys_scenes[scene_index]
    speech_loudness_level = speech_loudness_levels[scene_index]

    # Data to store for each participant
    grid_distributions = []
    path_efficiencies = []
    sweetspots = []

    for participant_id in range(25):
        pose_directory = f'./data/pose/participant{participant_id+1:02d}/'
        for file in os.listdir(pose_directory):
            if f'Arena {scene_index+1}.csv' in file:
                pose_file = f'{pose_directory}{file}'

        # Extract sweet spots and angles for each participant
        sweetspot, sweetangle = extract_sweetspot(pose_file)

        # Extract grid distribution
        grid_distribution = extract_grid_distribution(pose_file, x_distances=0.01, y_distances=0.01)

        # Extract path efficiency
        path_efficiency = extract_path_efficiency(pose_file)

        # Store data for each participant
        sweetspots.append(sweetspot)
        grid_distributions.append(grid_distribution)
        path_efficiencies.append(path_efficiency)

    # Convert lists to numpy arrays
    sweetspots = np.array(sweetspots)
    grid_distributions = np.array(grid_distributions)
    path_efficiencies = np.array(path_efficiencies)

    # Compute the average of the grid distributions across participants
    mean_grid_distribution = np.log10(np.mean(grid_distributions, axis=0) + 1e-10)

    # Initialize the plot
    fig = plt.figure()
    ax = plt.gca()
    plt.axis('scaled')

    # Plot the participant trajectories
    plt.imshow(mean_grid_distribution.T, origin='lower', extent=(-3.5, 3.5, -3.5, 3.5), cmap='gray_r')
    plt.colorbar(label='Time Spent (Log Probability)')
    plt.title(f'S{(scene_index%6)+1}-{cue_modality} Participant trajectories')
    plt.xlim([-3.0, 3.0])
    plt.ylim([-3.0, 3.2])
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')

    # Annotate the band locations
    source_locs = [(-1.5, 3.0), (-0.5, 3.0), (0.5, 3.0), (1.5, 3.0)]
    source_names = ['bass', 'drums', 'vocals', 'other']
    source_colors = ['tab:blue', 'tab:green', 'tab:cyan', 'tab:purple']

    # Annotate band locations
    for i in range(len(source_locs)):
        plt.scatter(source_locs[i][0], source_locs[i][1], color=source_colors[i], s=50, marker='o')
        plt.text(source_locs[i][0], source_locs[i][1] - 0.2, source_names[i], fontsize=10, ha='center', va='center', color='black')

    # Plot speech locations
    for i, speech_xy in enumerate(speech_xys_scene):
        plt.scatter(speech_xy[0], speech_xy[1], color='tab:red', s=50, marker='o')
        plt.text(speech_xy[0], speech_xy[1] + 0.2, f'{speech_loudness_level[i]} dB', fontsize=10, ha='center', va='center', color='black')
        plt.text(speech_xy[0], speech_xy[1] - 0.2, 'speech', fontsize=10, ha='center', va='center', color='black')

    # Annotate grid lines
    plt.vlines(x=[-1.5, -0.5, 0.5, 1.5], ymin=-2.5, ymax=2.5, color='black', linestyles='dashed', linewidth=0.5)
    plt.hlines(y=[-1.5, -0.5, 0.5, 1.5], xmin=-2.5, xmax=2.5, color='black', linestyles='dashed', linewidth=0.5)
    plt.vlines(x=[-2.5, 2.5], ymin=-2.5, ymax=2.5, color='black', linestyles='solid', linewidth=1.0)
    plt.hlines(y=[-2.5, 2.5], xmin=-2.5, xmax=2.5, color='black', linestyles='solid', linewidth=1.0)

    # Annotate the start point
    start_point = start_points[scene_index]
    start_marker = start_markers[scene_index]
    plt.scatter(start_point[0], start_point[1], color='green', s=100, marker=start_marker, label='Start point')
    plt.text(start_point[0], start_point[1] - 0.3, 'START', fontsize=10, ha='center', va='center', color='black')

    # Annotate the end points
    plt.scatter(sweetspots[:, 0], sweetspots[:, 1], s=10, marker='*', 
                color=modality_color, alpha=0.7, label='End points')

    plt.legend(loc='lower center', ncol=2, fontsize=8)
    # plt.show()
    plt.savefig(f'{plots_folder}S{(scene_index%6)+1}-{cue_modality}_grid_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # Store data for each scene
    path_efficiencies_per_scene.append(path_efficiencies)

path_efficiencies_per_scene = np.array(path_efficiencies_per_scene)

# Convert array into long data frame for plotting
df_list = []
for scene_index in range(12):
    modality = 'A' if scene_index < 6 else 'AV'
    environment_name = 'Auditory' if scene_index < 6 else 'Audio-Visual'
    scene_number = (scene_index % 6) + 1
    
    path_efficiencies = path_efficiencies_per_scene[scene_index][:]
    for i, path_efficiency in enumerate(path_efficiencies):
        df_list.append({'Participant': f'P{i+1:02d}',
                        'Path Efficiency': path_efficiency, 
                        'Scene': f'S{scene_number}',
                        'Modality': environment_name})

df = pd.DataFrame(df_list)

# Save data frame
df.to_csv(f'{plots_folder}navigation_results.csv', index=False)

# Load data frame
df = pd.read_csv(f'{plots_folder}navigation_results.csv')

# Plot box plot
fig, ax = plt.subplots(figsize=(8, 5))
# fig, ax = plt.subplots()
sns.boxplot(data=df, x='Scene', y='Path Efficiency', hue='Modality', 
            palette={'Auditory': 'steelblue', 'Audio-Visual': 'plum'}, 
            gap=.1, ax=ax)
plt.ylim([0, 1])
plt.title('Participant path efficiencies across scenes and modalities')
plt.savefig(f'{plots_folder}path_efficiency.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Run two-way repeated measures anova
anova_results = pg.rm_anova(
    data=df, 
    dv='Path Efficiency',
    within=['Scene', 'Modality'],
    subject='Participant',
    detailed=True,
)
print(anova_results)

# Run post-hoc tests
post_hocs = pg.pairwise_tests(
    data=df, 
    dv='Path Efficiency', 
    within='Modality',
    subject='Participant',
    alpha=0.5, 
    alternative='greater'
)
print(post_hocs)