import numpy as np
import os
import matplotlib.pyplot as plt
from src.utils import *

# Save to output folder
plots_folder = f'./output/4_subjective_maps/'
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

# Iterate over all scenes
for scene_index in range(12):
    cue_modality = 'A' if scene_index < 6 else 'AV'
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

    # Subjective scores map
    X, Y, Phi, Z = calculate_mahalanobis_map(sweetspots, sweetangles, grid_size=11, azi_res=5)

    fig, ax = plt.subplots()
    ax.axis('scaled')
    ax.set_title(f'S{(scene_index%6)+1}-{cue_modality} Subjective sweet spot map')
    ax.set_xlim([-3.0, 3.0])
    ax.set_ylim([-3.0, 3.2])
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')

    # Find the angle with the maximum Z value. 
    idx = np.argmax(Z)
    max_phi = Phi.ravel()[idx]
    _, _, phi_idx = np.unravel_index(idx, Z.shape)

    # Get contour based on X, Y, Z
    contour_set = ax.contourf(X[:, :, phi_idx], Y[:, :, phi_idx], Z[:, :, phi_idx], 
                              vmin=0, vmax=1, levels=np.linspace(0, 1, 11))
    fig.colorbar(contour_set, label = f'Normalized SQ Mahalanobis Score')
    ax = annotate_environment(ax, sweetspots, sweetangles, speech_xys_scene, speech_loudness_level, modality_color, label='Subjective sweet spots')

    # # Annotate a compass depending on angle
    ax = annotate_compass(ax, angle=max_phi)

    # Save figures
    ax.legend(loc='lower center', fontsize=8)
    plt.savefig(f'{plots_folder}S{(scene_index%6)+1}-{cue_modality}_subjective_map.pdf', dpi=300, bbox_inches='tight')
    plt.close()