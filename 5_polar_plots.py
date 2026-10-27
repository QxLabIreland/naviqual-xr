import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from src.utils import *

# Save to output folder
plots_folder = f'./output/5_polar_plots/'
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

# Iterate over auditory scenes
for scene_index in range(6):
    print(f'Analysing scene S{(scene_index%6)+1}...')
    speech_xys_scene = speech_xys_scenes[scene_index]
    speech_loudness_level = speech_loudness_levels[scene_index]

    # Obtain naviqual data
    naviqual_file = f'./data/naviqual/scene{(scene_index%6)+1}.csv'
    df_naviqual = pd.read_csv(naviqual_file)

    # Data to store for each participant
    ad_sweetspots = []
    ad_sweetangles = []
    av_sweetspots = []
    av_sweetangles = []

    for participant_id in range(25):
        pose_directory = f'./data/pose/participant{participant_id+1:02d}/'
        for file in os.listdir(pose_directory):
            if f'Arena {scene_index+1}.csv' in file:
                ad_pose_file = f'{pose_directory}{file}'
            elif f'Arena {scene_index+7}.csv' in file:
                av_pose_file = f'{pose_directory}{file}'

        # Extract sweet spot and sweet angle
        ad_sweetspot, ad_sweetangle = extract_sweetspot(ad_pose_file)
        av_sweetspot, av_sweetangle = extract_sweetspot(av_pose_file)

        # Store data for each participant
        ad_sweetspots.append(ad_sweetspot)
        ad_sweetangles.append(ad_sweetangle)
        av_sweetspots.append(av_sweetspot)
        av_sweetangles.append(av_sweetangle)

    # Convert lists to numpy arrays
    ad_sweetspots = np.array(ad_sweetspots)
    ad_sweetangles = np.array(ad_sweetangles)
    av_sweetspots = np.array(av_sweetspots)
    av_sweetangles = np.array(av_sweetangles)

    # Subjective scores map
    X_subj_ad, Y_subj_ad, Phi_subj_ad, Z_subj_ad = calculate_mahalanobis_map(ad_sweetspots, ad_sweetangles, grid_size=11, azi_res=5)
    X_subj_av, Y_subj_av, Phi_subj_av, Z_subj_av = calculate_mahalanobis_map(av_sweetspots, av_sweetangles, grid_size=11, azi_res=5)

    # Objective scores map
    X_snr, Y_snr, Phi_snr, Z_snr = extract_objective_mahalanobis(df_naviqual, 'snr', sweetspot_threshold=0.8)
    X_locq, Y_locq, Phi_locq, Z_locq = extract_objective_mahalanobis(df_naviqual, 'locq', sweetspot_threshold=0.8)

    mesh_dict = {
        'subj_ad': (X_subj_ad, Y_subj_ad, Phi_subj_ad, Z_subj_ad), 
        'subj_av': (X_subj_av, Y_subj_av, Phi_subj_av, Z_subj_av),
        'snr': (X_snr, Y_snr, Phi_snr, Z_snr), 
        'locq': (X_locq, Y_locq, Phi_locq, Z_locq)
    }

    # POLAR PLOTS
    # Find grid location with max Z_subj_av
    idx = np.argmax(Z_subj_av)
    x_loc = X_subj_av.ravel()[idx]
    y_loc = Y_subj_av.ravel()[idx]
    for response_name in ['snr', 'locq']:
        response_name_title = 'LQ' if response_name == 'locq' else 'SNR'

        # Find subset with fixed location
        p_subj_ad, z_subj_ad = fixed_location_polar(x_loc, y_loc, Phi_subj_ad, Z_subj_ad)
        p_subj_av, z_subj_av = fixed_location_polar(x_loc, y_loc, Phi_subj_av, Z_subj_av)
        p_snr, z_snr = fixed_location_polar(x_loc, y_loc, Phi_snr, Z_snr)
        p_locq, z_locq = fixed_location_polar(x_loc, y_loc, Phi_locq, Z_locq)

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(np.deg2rad(p_subj_ad), z_subj_ad, label='SQ-A', color='blue')
        ax.plot(np.deg2rad(p_subj_av), z_subj_av, label='SQ-AV', color='magenta')
        ax.plot(np.deg2rad(p_snr), z_snr, label='SNR', color='red')
        ax.plot(np.deg2rad(p_locq), z_locq, label='LQ', color='green')
        ax.set_rmax(1)
        ax.set_rticks([0, 0.5, 1])
        ax.set_rlabel_position(135)
        ax.set_theta_zero_location('N')
        ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
        ax.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '-135°', '-90°', '-45°'])
        ax.set_yticklabels([])
        ax.grid(True)
        ax.legend()

        # Annotate the virtual environment on lower right
        ax = environment_on_polar(ax, (x_loc, y_loc), speech_xys_scene)

        plt.savefig(f'{plots_folder}S{(scene_index%6)+1}_polar.jpg', dpi=300, bbox_inches='tight')
        plt.close()