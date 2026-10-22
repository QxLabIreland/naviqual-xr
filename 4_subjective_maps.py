import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
    X_subj, Y_subj, Phi_subj, Z_subj = calculate_mahalanobis_map(sweetspots, sweetangles, grid_size=11, azi_res=5)

    # ANIMATIONS
    fig, ax = plt.subplots()

    # Define update function for animations
    def update(frame):
        current_angle = Phi_subj[0, 0, frame]
        ax.clear()
        contour_quad1 = ax.contourf(X_subj[:, :, frame], Y_subj[:, :, frame], Z_subj[:, :, frame], 
                                cmap='viridis', vmin=0, vmax=1, levels=10)
        fig.suptitle(f'Direction: {current_angle:.0f} degrees')
        ax.axis('scaled')
        ax.set_title('Subjective Sweet Spot Map')

        # Annotate environment
        annotate_environment(ax, sweetspots, sweetangles, speech_xys_scene, speech_loudness_level, modality_color)

        # Annotate a compass depending on angle
        annotate_compass(ax, angle=current_angle)
        return contour_quad1.collections

    # Create animation
    frames = np.arange(np.shape(Phi_subj)[2])
    anim = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

    # Save animation as GIF
    anim.save(f'{plots_folder}S{(scene_index%6)+1}-{cue_modality}_subjective_map.gif', 
              writer='pillow', fps=10)
    plt.close()