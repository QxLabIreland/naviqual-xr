import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from src.utils import *

# Save to output folder
plots_folder = f'./output/5_objective_maps/'
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

    # Obtain naviqual data
    naviqual_file = f'./data/naviqual/scene{(scene_index%6)+1}.csv'
    df_naviqual = pd.read_csv(naviqual_file)

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

    # Objective scores map
    X_snr, Y_snr, Phi_snr, Z_snr = extract_objective_mahalanobis(df_naviqual, 'snr', sweetspot_threshold=0.8)
    X_locq, Y_locq, Phi_locq, Z_locq = extract_objective_mahalanobis(df_naviqual, 'locq', sweetspot_threshold=0.8)

    mesh_dict = {
        'snr': (X_snr, Y_snr, Phi_snr, Z_snr), 
        'locq': (X_locq, Y_locq, Phi_locq, Z_locq)
    }

    # Create objective sweet spot maps with subjective sweet spot data points
    for response_name in ['snr', 'locq']:
        response_name_title = 'LQ' if response_name == 'locq' else 'SNR'
        X, Y, Phi, Z = mesh_dict[response_name]
        fig, ax = plt.subplots()

        # Define update function for animations
        def update(frame):
            current_angle = Phi[0, 0, frame]
            ax.clear()
            contour_quad1 = ax.contourf(X[:, :, frame], Y[:, :, frame], Z[:, :, frame], 
                                    cmap='viridis', vmin=0, vmax=1, levels=10)
            fig.suptitle(f'Direction: {current_angle:.0f} degrees')
            ax.axis('scaled')
            ax.set_title(f'{response_name_title} Sweet Spot Map')

            # Annotate environment
            annotate_environment(ax, sweetspots, sweetangles, speech_xys_scene, speech_loudness_level, modality_color)

            # Annotate a compass depending on angle
            annotate_compass(ax, angle=current_angle)
            return contour_quad1.collections

        # Create animation
        frames = np.arange(np.shape(Phi)[2])
        anim = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

        # Save animation as GIF
        anim.save(f'{plots_folder}S{(scene_index%6)+1}-{cue_modality}_{response_name}_ss_map.gif', 
                writer='pillow', fps=10)
        plt.close()