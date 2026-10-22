import numpy as np
import os
import pandas as pd
from src.utils import *

# Save to output folder
output_folder = f'./output/6_similarity_metrics/'
os.makedirs(output_folder, exist_ok=True)

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
mae_scores_snr = np.zeros(12)
mae_scores_locq = np.zeros(12)
ssim_scores_snr = np.zeros(12)
ssim_scores_locq = np.zeros(12)
emd_scores_snr = np.zeros(12)
emd_scores_locq = np.zeros(12)

# Iterate over all scenes
for scene_index in range(12):
    print(f'Analysing scene S{scene_index+1}...')
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

    # Compute mean absolute error
    mae_scores_snr[scene_index] = np.mean(np.abs(Z_subj - Z_snr))
    print(f'Mean Absolute Error Score SNR: {mae_scores_snr[scene_index]:.4f}')
    mae_scores_locq[scene_index] = np.mean(np.abs(Z_subj - Z_locq))
    print(f'Mean Absolute Error Score LQ: {mae_scores_locq[scene_index]:.4f}')

    # Compute SSIM
    ssim_scores_snr[scene_index] = compute_ssim(Z_subj, Z_snr, win_size=3)
    print(f'SSIM Score SNR: {ssim_scores_snr[scene_index]:.4f}')
    ssim_scores_locq[scene_index] = compute_ssim(Z_subj, Z_locq, win_size=3)
    print(f'SSIM Score LQ: {ssim_scores_locq[scene_index]:.4f}')

    # Compute EMD
    emd_scores_snr[scene_index] = compute_emd(Z_subj, Z_snr)
    print(f'EMD Score SNR: {emd_scores_snr[scene_index]:.4f}')
    emd_scores_locq[scene_index] = compute_emd(Z_subj, Z_locq)
    print(f'EMD Score LQ: {emd_scores_locq[scene_index]:.4f}')

# Create a data frame and save results to a csv file
cue_modalities = ['A'] * 6 + ['AV'] * 6
df_scene_results = pd.DataFrame({
    'scene_id': [f'S{(i%6)+1}-{cue_modality}' for i, cue_modality in enumerate(cue_modalities)], 
    'mae_score_snr': [f'{score:.4f}' for score in mae_scores_snr],
    'mae_score_locq': [f'{score:.4f}' for score in mae_scores_locq],
    'ssim_score_snr': [f'{score:.4f}' for score in ssim_scores_snr],
    'ssim_score_locq': [f'{score:.4f}' for score in ssim_scores_locq],
    'emd_score_snr': [f'{score:.4f}' for score in emd_scores_snr],
    'emd_score_locq': [f'{score:.4f}' for score in emd_scores_locq],
})
df_scene_results.to_csv(f'{output_folder}similarity_metrics.csv', index=False)