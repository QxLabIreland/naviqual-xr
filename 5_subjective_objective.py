import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from src.utils import *

# Save to output folder
plots_folder = f'./output/5_subjective_objective/'
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

    # Subjective scores map
    X_subj, Y_subj, Phi_subj, Z_subj = calculate_mahalanobis_map(sweetspots, sweetangles, grid_size=11, azi_res=5)

    # Objective scores map
    X_snr, Y_snr, Phi_snr, Z_snr = extract_objective_mahalanobis(df_naviqual, 'snr', sweetspot_threshold=0.8)
    X_locq, Y_locq, Phi_locq, Z_locq = extract_objective_mahalanobis(df_naviqual, 'locq', sweetspot_threshold=0.8)

    # ANIMATIONS
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Define update function
    def update(frame):
        current_angle = Phi_subj[0, 0, frame]
        for ax in axs:
            ax.clear()
        contour_quad1 = axs[0].contourf(X_subj[:, :, frame], Y_subj[:, :, frame], Z_subj[:, :, frame], 
                                cmap='viridis', vmin=0, vmax=1, levels=10)
        axs[0].set_title('Subjective Sweet Spot Map')
        contour_quad2 = axs[1].contourf(X_snr[:, :, frame], Y_snr[:, :, frame], Z_snr[:, :, frame], 
                                cmap='viridis', vmin=0, vmax=1, levels=10)
        axs[1].set_title('SNR Sweet Spot Map')
        contour_quad3 = axs[2].contourf(X_locq[:, :, frame], Y_locq[:, :, frame], Z_locq[:, :, frame], 
                                cmap='viridis', vmin=0, vmax=1, levels=10)
        axs[2].set_title('LQ Sweet Spot Map')  
        fig.suptitle(f'Direction: {current_angle:.0f} degrees')

        for ax in axs:
            # Annotate environment
            annotate_environment(ax, sweetspots, sweetangles, speech_xys_scene, speech_loudness_level, modality_color)

            # Annotate compass depending on angle
            annotate_compass(ax, angle=current_angle)

        return contour_quad1.collections + contour_quad2.collections + contour_quad3.collections

    # Create animation
    frames = np.arange(np.shape(Phi_subj)[2])
    anim = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

    # Save animation as GIF
    anim.save(f'{plots_folder}S{(scene_index%6)+1}-{cue_modality}_subjective_objective.gif', 
            writer='pillow', fps=10)
    plt.close()